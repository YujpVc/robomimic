"""
This file contains the Fairino environment wrapper.
It is decoupled from serl_robot_infra and uses the local Fairino_Arm driver.
Refactored to use a background servo thread for smooth control, mirroring
the logic from fairino_server.py.

UPDATES:
- Integrated DH_Gripper support (threaded) for robust gripper control.
- Improved RealSenseCamera robustness.
- Standardized action scaling and safety checks.
"""
import sys
import os
import time
import threading
import numpy as np
import gymnasium as gym
from copy import deepcopy
from scipy.spatial.transform import Rotation as R, Slerp
from collections import OrderedDict
from typing import Optional, Tuple, Any
import serial
import binascii

# Add local Fairino_Arm to path to ensure we can import the driver
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FAIRINO_LIB_PATH = os.path.join(CURRENT_DIR, "Fairino_Arm")
if FAIRINO_LIB_PATH not in sys.path:
    sys.path.append(FAIRINO_LIB_PATH)

# Import RPC directly from the local library
from fairino.Robot import RPC, RobotError # type: ignore

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils
import pyrealsense2 as rs


# -----------------------------------------------------------------------------
# DH Gripper Driver (Embedded for portability)
# -----------------------------------------------------------------------------
class DH_Gripper:
    def __init__(self, device='/dev/dh_gripper', baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=1):
        self.device = device
        self.baudrate = baudrate
        self.ser = None
        self._connect()

    def _connect(self):
        # Allow error to propagate - do not use try/except
        self.ser = serial.Serial(self.device, baudrate=self.baudrate, bytesize=8, parity='N', stopbits=1, timeout=1)
        print(f"DH_Gripper connected on {self.device}")

    def calculate_crc(self, data):
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for i in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc.to_bytes(2, byteorder='little')

    def send_command(self, command):
        if self.ser is None: return b''
        # Allow error to propagate - do not use try/except
        crc = self.calculate_crc(binascii.unhexlify(command))
        command = command + binascii.hexlify(crc).decode('utf-8')
        self.ser.write(binascii.unhexlify(command))
        response = self.ser.read(8)
        return response

    def initialize_gripper(self):
        # mode = 0: default init
        init_command = '010601000001'
        return self.send_command(init_command)

    def get_initialization_status(self):
        status_command = '010302000001'
        response = self.send_command(status_command)
        if response == b'': return -1
        try:
            response = binascii.hexlify(response).decode('utf-8').split('010302')[1]
            return int(response[:4], 16)
        except:
            return -1

    def set_gripper_position(self, position):
        position = int(np.clip(position, 0, 1000))
        position_command = '01060103' + format(position, '04X')
        return self.send_command(position_command)
    
    def get_current_position(self):
        position_command = '010302020001'
        response = self.send_command(position_command)
        if response == b'': return -1
        try:
            response = binascii.hexlify(response).decode('utf-8').split('010302')[1]
            return int(response[:4], 16)
        except:
            return -1


class GripperController:
    """Threaded wrapper for DH_Gripper to prevent blocking main loops."""
    def __init__(self, device='/dev/dh_gripper'):
        self.device = device
        self.gripper = DH_Gripper(device=device)
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        
        # State
        self.current_pos = 1000 # Assume open
        self.target_pos = 1000
        self.last_cmd_time = 0
        self.cmd_interval = 0.05 # Limit write freq
        
        # Initialization
        self._init_gripper()

    def _init_gripper(self):
        if self.gripper.ser:
            self.gripper.initialize_gripper()
            time.sleep(1.0)
            status = self.gripper.get_initialization_status()
            print(f"Gripper Init Status: {status}")
            # Init to closed (0)
            self.gripper.set_gripper_position(0)
            
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def set_target(self, pos_0_1000):
        with self.lock:
            self.target_pos = int(pos_0_1000)

    def get_position(self):
        with self.lock:
            return self.current_pos

    def _update_loop(self):
        while self.running:
            # 1. Write Target (if changed or periodic?)
            # We only write if needed to save bandwidth, or periodically?
            # DH gripper needs explicit set commands.
            with self.lock:
                tgt = self.target_pos
            
            # Simple logic: Always send target at low freq
            if time.time() - self.last_cmd_time > self.cmd_interval:
                self.gripper.set_gripper_position(tgt)
                self.last_cmd_time = time.time()
            
            # 2. Read State
            pos = self.gripper.get_current_position()
            if pos != -1:
                with self.lock:
                    self.current_pos = pos
            
            time.sleep(0.02) # 50Hz polling


# -----------------------------------------------------------------------------
# RealSense Camera Wrapper
# -----------------------------------------------------------------------------
class RealSenseCamera:
    """Wrapper for RealSense camera."""
    def __init__(self, serial_number=None, width=640, height=480, fps=30):
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if serial_number:
            self.config.enable_device(serial_number)
            
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        
        # Allow error to propagate
        self.profile = self.pipeline.start(self.config)
        print(f"RealSense camera {serial_number if serial_number else '(auto)'} started.")

        self.latest_frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if not self.pipeline:
                time.sleep(0.1)
                continue
            # Allow error to propagate
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
                with self.lock:
                    self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None


def _euler_2_quat(euler_xyz: np.ndarray) -> np.ndarray:
    """Euler xyz (rad) -> quat xyzw."""
    return R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64)).as_quat()

def _maybe_call(robot: Any, name: str, *args, **kwargs):
    """Safely call a robot method if it exists."""
    if robot is None: return None
    fn = getattr(robot, name, None)
    if fn is None:
        return None
    return fn(*args, **kwargs)

class DefaultEnvConfig:
    """Default configuration for FairinoEnv."""
    ROBOT_IP: str = "192.168.58.6"
    
    # Servo parameters
    CMD_T: float = 0.008  # 8ms servo period (125Hz)
    FILTER_T: float = 0.04 # Filter time constant

    # Camera Config
    REALSENSE_CAMERAS = {
        'agentview_image': '043422250358', 
        'robot0_eye_in_hand_image': '140122073169' 
    }
    IMAGE_CROP = {}

    TARGET_POSE = np.zeros((6,))
    REWARD_THRESHOLD = np.zeros((6,))
    ACTION_SCALE = np.array([0.5, 0.5, 1.0])
    
    # Action range limits for normalization [-1, 1] -> actual range
    # Position: maximum delta per step (in meters)
    MAX_POS_DELTA = 0.01  # 1cm per step (at 20Hz = 0.2m/s)
    # Rotation: maximum delta per step (in radians)
    MAX_ROT_DELTA = 0.04   # ~2.2 degrees per step
    
    # Default reset pose
    RESET_POSE = np.zeros((6,))
    # Joint reset target (degrees)
    RESET_JOINT_TARGET = [-90.0, -90.0, -90.0, -90.0, 90.0, 0.0]

    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)

    DISPLAY_IMAGE = False
    GRIPPER_SLEEP = 1.0
    MAX_EPISODE_LENGTH = 100
    JOINT_RESET_PERIOD = 0


class FairinoServoController:
    """
    Background thread controller for Fairino Robot + DH Gripper.
    """
    def __init__(self, ip, config: DefaultEnvConfig, fake_env=False):
        self.ip = ip
        self.config = config
        self.fake_env = fake_env
        self.robot = None
        
        # Cameras
        self.cameras = {}
        if not self.fake_env and config.REALSENSE_CAMERAS:
            for cam_name, serial in config.REALSENSE_CAMERAS.items():
                self.cameras[cam_name] = RealSenseCamera(serial_number=serial)
        
        # Gripper (Threaded)
        self.gripper_controller = None
        if not self.fake_env:
            self.gripper_controller = GripperController()

        # State
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self._target_desc_pos = None # xyz (mm) + rpy (deg)
        
        # Cached observation state
        self.curr_pose_7d = np.zeros(7)
        self.curr_q_deg = np.zeros(6)
        self.curr_gripper_pos = 1.0 # Default Open
        self.curr_images = {}
        
        # [Fix Jitter] Store last quaternion for continuity check
        self.last_quat = None

        if not self.fake_env:
            # Allow error to propagate
            self.robot = RPC(ip=self.ip)
            print(f"Connected to Fairino Robot at {self.ip}")

    def start(self):
        if self.fake_env or self.thread is not None:
            return
        self.running = True
        
        if self.gripper_controller:
            self.gripper_controller.start()

        self.thread = threading.Thread(target=self._servo_loop, daemon=True)
        self.thread.start()

    def stop(self, close_cameras=True):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        
        _maybe_call(self.robot, "ServoMoveEnd")
        
        if self.gripper_controller:
            self.gripper_controller.stop()

        if close_cameras:
            for cam in self.cameras.values():
                cam.close()

    def set_target_pose(self, pose_7d: np.ndarray):
        """Set the target pose (xyz m + quat xyzw)."""
        pose_7d = np.asarray(pose_7d, dtype=np.float64)
        xyz_mm = pose_7d[:3] * 1000.0
        rpy_deg = R.from_quat(pose_7d[3:]).as_euler("xyz", degrees=True)
        desc_pos = np.concatenate([xyz_mm, rpy_deg])

        with self.lock:
            self._target_desc_pos = desc_pos

    def set_gripper(self, open_amount: float):
        """Set gripper open amount 0.0 (closed) to 1.0 (open)."""
        if self.fake_env:
            self.curr_gripper_pos = open_amount
            return
            
        if self.gripper_controller:
            # Map 0.0-1.0 -> 0-1000
            target = open_amount * 1000.0
            self.gripper_controller.set_target(target)

    def get_state(self):
        """Fetch latest state from robot or cache."""
        if self.fake_env:
            with self.lock:
                return self.curr_pose_7d.copy(), self.curr_q_deg.copy(), self.curr_gripper_pos, {}
        
        # 1. Pose
        res_p = _maybe_call(self.robot, "GetActualTCPPose") # Default flag=1 (non-blocking in some SDKs)
        if res_p and res_p[0] == 0:
            pose6 = np.array(res_p[1], dtype=np.float64)
            xyz_m = pose6[:3] / 1000.0
            
            # [ROLLBACK] Fairino uses Euler RPY (degrees)
            # Servo error 14 confirmed rotvec was wrong interpretation
            quat = R.from_euler("xyz", pose6[3:], degrees=True).as_quat()
            
            # [Fix Jitter] Quaternion continuity (Double Cover)
            # If dot(q_curr, q_last) < 0, then q_curr and -q_curr represent same rotation
            # Flip q_curr to be close to q_last to prevent "jumping"
            if self.last_quat is not None:
                if np.dot(quat, self.last_quat) < 0:
                    quat = -quat
            self.last_quat = quat
            
            self.curr_pose_7d = np.concatenate([xyz_m, quat])
        
        # 2. Joints
        res_q = _maybe_call(self.robot, "GetActualJointPosDegree", 0)
        if res_q and res_q[0] == 0:
            self.curr_q_deg = np.array(res_q[1], dtype=np.float64)

        # 3. Gripper
        if self.gripper_controller:
            g_val = self.gripper_controller.get_position()
            if g_val != -1:
                self.curr_gripper_pos = g_val / 1000.0

        # 4. Images
        for name, cam in self.cameras.items():
            img = cam.get_frame()
            if img is not None:
                if name in self.config.IMAGE_CROP:
                    cx, cy, h, w = self.config.IMAGE_CROP[name]
                    img = img[cy:cy+h, cx:cx+w]
                self.curr_images[name] = img

        return self.curr_pose_7d.copy(), self.curr_q_deg.copy(), self.curr_gripper_pos, self.curr_images.copy()

    def _servo_loop(self):
        """High frequency control loop."""
        # Ensure servo mode is started
        _maybe_call(self.robot, "ServoMoveStart")
        # Give it a moment to initialize
        time.sleep(0.5)
        
        while self.running:
            t0 = time.time()
            
            with self.lock:
                target = self._target_desc_pos.copy() if self._target_desc_pos is not None else None
            
            if target is not None:
                # Debug: Ensure target is valid
                # print(f"ServoCart: {target}")
                ret = _maybe_call(
                    self.robot, 
                    "ServoCart",
                    0, # mode
                    target.tolist(),
                    [1.0]*6, # pos_gain
                    0.0, # acc
                    0.5, # vel (Reduced from 3.0 for safety/debugging)
                    self.config.CMD_T, # cmdT
                    self.config.FILTER_T, # filterT
                    0.0 # gain
                )
                if ret != 0:
                    print(f"[ServoError] Ret: {ret}")
            
            dt = time.time() - t0
            sleep_time = max(0.0, self.config.CMD_T - dt)
            time.sleep(sleep_time)

    def joint_reset(self, target_joints_deg):
        """Pause servo, move joints, resume servo."""
        if self.fake_env:
            return

        # Pause threads
        self.stop(close_cameras=False)
        time.sleep(0.2)

        _maybe_call(self.robot, "ResetAllError")
        
        # MoveJ
        print(f"Resetting joints to {target_joints_deg}")
        ret = _maybe_call(
            self.robot,
            "MoveJ",
            list(map(float, target_joints_deg)),
            0, # tool
            0, # user
            desc_pos=[0.0]*6,
            vel=20.0,
            acc=0.0,
            ovl=100.0,
            blendT=-1.0,
            offset_flag=0,
            offset_pos=[0.0]*6
        )
        
        # Wait for completion
        t_start = time.time()
        while True:
            res_q = _maybe_call(self.robot, "GetActualJointPosDegree", 0)
            if res_q is not None:
                if isinstance(res_q, (list, tuple)) and len(res_q) >= 2 and res_q[0] == 0:
                    curr_q = np.array(res_q[1], dtype=np.float64)
                    target_q = np.array(target_joints_deg, dtype=np.float64)
                    max_error = np.max(np.abs(curr_q - target_q))
                    if max_error < 1.0:
                        break
            
            if time.time() - t_start > 15.0:
                print(f"Warning: joint_reset timed out.")
                break
            time.sleep(0.1)
            
        time.sleep(0.5)

        with self.lock:
            self._target_desc_pos = None
            # Reset last_quat to None to restart continuity check on new episode
            self.last_quat = None

        # Restart
        self.start()

class EnvFairino(EB.EnvBase):
    """Wrapper class for Fairino environment to Robomimic API."""
    
    def __init__(
        self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        postprocess_visual_obs=True,
        **kwargs,
    ):
        self.postprocess_visual_obs = postprocess_visual_obs
        self._env_name = env_name
        self._init_kwargs = deepcopy(kwargs)
        
        config = DefaultEnvConfig()
        if "robot_ip" in kwargs:
            config.ROBOT_IP = kwargs["robot_ip"]

        self.fake_env = kwargs.get("fake_env", False)
        self.config = config
        
        self.hz = 20 # Default
        self.max_episode_length = int(config.MAX_EPISODE_LENGTH)
        
        # Action normalization ranges
        self.max_pos_delta = float(config.MAX_POS_DELTA)
        self.max_rot_delta = float(config.MAX_ROT_DELTA)

        self.controller = FairinoServoController(
            config.ROBOT_IP, 
            config, 
            fake_env=self.fake_env
        )

        # Initial State
        self.resetpos = np.concatenate([config.RESET_POSE[:3], _euler_2_quat(config.RESET_POSE[3:])])
        self.currpos = self.resetpos.copy()

        if not self.fake_env:
            self.controller.start()
        
        self.cycle_count = 0
        self.curr_path_length = 0
        
        self._current_obs = None

    def step(self, action):
        start_time = time.time()
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        
        # 1. Denormalize Pose Action
        xyz_delta = action[:3] * self.max_pos_delta
        rot_delta = action[3:6] * self.max_rot_delta

        # 2. Compute Next Pose
        nextpos = self.currpos.copy()
        nextpos[:3] = nextpos[:3] + xyz_delta
        nextpos[3:] = (R.from_euler("xyz", rot_delta) * R.from_quat(self.currpos[3:])).as_quat()

        # 3. Update Controller Target
        self.controller.set_target_pose(nextpos)
        
        # 4. Handle Gripper (Action -1 to 1)
        gripper_action = float(action[6])
        gripper_open_amount = (gripper_action + 1.0) / 2.0
        # Optional: Thresholding for binary gripper
        if gripper_open_amount > 0.5: gripper_open_amount = 1.0
        else: gripper_open_amount = 0.0
        
        self.controller.set_gripper(gripper_open_amount)
        
        # 5. Update internal state
        self.currpos = nextpos
        self.curr_path_length += 1
        
        # 6. Wait for Hz
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        # 7. Get Obs
        di = self._get_obs()
        self._current_obs = di
        
        done = self.curr_path_length >= self.max_episode_length
        reward = 0
        info = {"succeed": False}
        
        return self.get_observation(di), reward, self.is_done(), info

    def reset(self, joint_reset=None):
        # Default logic for reset: joint reset on period
        self.cycle_count += 1
        
        if joint_reset is None:
            joint_reset = False
            if self.config.JOINT_RESET_PERIOD != 0 and self.cycle_count % self.config.JOINT_RESET_PERIOD == 0:
                joint_reset = True
            
        if self.fake_env:
            self.currpos = self.resetpos.copy()
            di = self._get_obs()
            self._current_obs = di
            return self.get_observation(di)

        if joint_reset:
            self.controller.joint_reset(self.config.RESET_JOINT_TARGET)
        else:
            self.controller.set_target_pose(self.resetpos)
            time.sleep(2.0)
        
        # Close gripper on reset (Default state)
        self.controller.set_gripper(0.0)
        
        input(f"Please replace object and press Enter...")
        
        # Sync state
        self.currpos, _, _, _ = self.controller.get_state()
        self.curr_path_length = 0
        
        di = self._get_obs()
        self._current_obs = di
        return self.get_observation(di)
    
    def _get_obs(self) -> dict:
        pose_7d, q_deg, g_pos, images = self.controller.get_state()
        
        # Re-sync current pose tracking to actual to prevent drift
        self.currpos = pose_7d
        
        # Map gripper 0-1 -> -1 to 1
        g_action_space = g_pos * 2.0 - 1.0
        
        return {
            "state": {
                "tcp_pose": pose_7d,
                "gripper_pose": np.array([g_action_space]),
                "joint_pose": np.deg2rad(q_deg)
            },
            "images": images
        }

    def close(self):
        self.controller.stop()

    def get_observation(self, di=None):
        if di is None:
            di = self._current_obs
            if di is None:
                di = self._get_obs()
        
        ret = {}
        if "state" in di:
            state = di["state"]
            if "tcp_pose" in state:
                tcp_pose = state["tcp_pose"]
                ret["robot0_eef_pos"] = np.array(tcp_pose[:3])
                ret["robot0_eef_quat"] = np.array(tcp_pose[3:])
            if "gripper_pose" in state:
                ret["robot0_gripper_qpos"] = np.array(state["gripper_pose"])
            if "joint_pose" in state:
                ret["robot0_joint_pos"] = np.array(state["joint_pose"])
                ret["robot0_joint_pos_cos"] = np.cos(ret["robot0_joint_pos"])
                ret["robot0_joint_pos_sin"] = np.sin(ret["robot0_joint_pos"])

        if "images" in di:
             for k, v in di["images"].items():
                key = k if k.endswith("_image") else f"{k}_image"
                ret[key] = v
                if self.postprocess_visual_obs:
                     # Manually process if modalities not initialized
                     if ObsUtils.OBS_KEYS_TO_MODALITIES is None:
                         # Default processing for images: HWC -> CHW, 0-1 float
                         if len(v.shape) == 3 and v.shape[2] == 3: # RGB
                             ret[key] = ObsUtils.process_frame(v, 3, 255.)
                         elif len(v.shape) == 3 and v.shape[2] == 1: # Depth
                             ret[key] = ObsUtils.process_frame(v, 1, 1.)
                         else:
                             ret[key] = v # Unknown, keep as is
                     else:
                         ret[key] = ObsUtils.process_obs(obs=ret[key], obs_key=key)
        return ret

    def get_state(self):
        """Get environment simulator state.
        For real robot, we return the current observation as state representation.
        """
        return self._current_obs

    def reset_to(self, state):
        """
        Reset to a specific simulator state.
        For real robot, this is not fully supported. We treat it as a reset.
        """
        # Warning: Cannot reset real robot to arbitrary state
        return self.reset()

    def render(self, mode="human", height=None, width=None, camera_name=None):
        """Render"""
        # If camera_name is provided, return that camera image
        if camera_name is not None:
             if self._current_obs and "images" in self._current_obs and camera_name in self._current_obs["images"]:
                 return self._current_obs["images"][camera_name]
        return None

    def get_reward(self):
        """
        Get current reward.
        """
        return 0.0

    def get_goal(self):
        """
        Get goal observation. 
        """
        return None

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification.
        """
        pass

    def is_done(self):
        return False

    def is_success(self):
        return {"task": False}

    @property
    def action_dimension(self):
        return 7

    @property
    def name(self):
        return self._env_name

    @property
    def type(self):
        return 5 

    def serialize(self):
        return dict(
            env_name=self.name,
            type=self.type,
            env_kwargs=deepcopy(self._init_kwargs)
        )

    @classmethod
    def create_for_data_processing(cls, env_name, camera_names, **kwargs):
        return cls(env_name=env_name, use_image_obs=(len(camera_names)>0), **kwargs)

    @property
    def rollout_exceptions(self):
        return ()
