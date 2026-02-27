"""
This file contains the Fairino environment wrapper.
It is decoupled from serl_robot_infra and uses the local Fairino_Arm driver.
Refactored to use a background servo thread for smooth control, mirroring
the logic from fairino_server.py.
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

# Add local Fairino_Arm to path to ensure we can import the driver
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FAIRINO_LIB_PATH = os.path.join(CURRENT_DIR, "Fairino_Arm")
if FAIRINO_LIB_PATH not in sys.path:
    sys.path.append(FAIRINO_LIB_PATH)

# Import RPC directly from the local library
from fairino.Robot import RPC, RobotError
import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils
import pyrealsense2 as rs


class RealSenseCamera:
    """Wrapper for RealSense camera."""
    def __init__(self, serial_number=None, width=640, height=480, fps=30):
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        if serial_number:
            self.config.enable_device(serial_number)
            
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        
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
            self.pipeline.stop()
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
    # Format: { 'camera_name': 'serial_number' }
    # Use empty serial for auto-detection of first camera if only one.
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
    MAX_POS_DELTA = 0.01  # 2.5cm per step (缩小到原来的1/2)
    # Rotation: maximum delta per step (in radians)
    MAX_ROT_DELTA = 0.02   # ~2.86 degrees per step (缩小到原来的1/2)
    
    # Default reset pose (rad for orientation if using euler)
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
    Background thread controller for Fairino Robot.
    Maintains high-frequency control loop separate from the env step.
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
        
        # State
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self._target_pose_7d = None # xyz (m) + quat (xyzw)
        self._target_desc_pos = None # xyz (mm) + rpy (deg)
        
        # Cached observation state
        self.curr_pose_7d = np.zeros(7)
        self.curr_q_deg = np.zeros(6)
        self.curr_gripper_pos = 0.0
        self.curr_images = {}

        if not self.fake_env:
            try:
                self.robot = RPC(ip=self.ip)
                print(f"Connected to Fairino Robot at {self.ip}")
            except Exception as e:
                print(f"Failed to connect to robot: {e}")
                # Don't raise, allow running in degraded state or handle upstream
                pass

    def start(self):
        if self.fake_env or self.thread is not None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._servo_loop, daemon=True)
        self.thread.start()

    def stop(self, close_cameras=True):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        _maybe_call(self.robot, "ServoMoveEnd")
        
        # Stop cameras
        if close_cameras:
            for cam in self.cameras.values():
                cam.close()

    def set_target_pose(self, pose_7d: np.ndarray):
        """Set the target pose (xyz m + quat xyzw)."""
        pose_7d = np.asarray(pose_7d, dtype=np.float64)
        
        # Convert to Fairino format: xyz (mm) + rpy (deg)
        xyz_mm = pose_7d[:3] * 1000.0
        rpy_deg = R.from_quat(pose_7d[3:]).as_euler("xyz", degrees=True)
        desc_pos = np.concatenate([xyz_mm, rpy_deg])

        with self.lock:
            self._target_pose_7d = pose_7d
            self._target_desc_pos = desc_pos

    def get_state(self):
        """Fetch latest state from robot or cache."""
        if self.fake_env:
            with self.lock:
                return self.curr_pose_7d.copy(), self.curr_q_deg.copy(), self.curr_gripper_pos, {}
        
        # This blocks briefly, so we might want to do this in the loop or separate thread
        # For simplicity, we query directly but use maybe_call
        
        # 1. Pose
        res_p = _maybe_call(self.robot, "GetActualTCPPose", 0) # 0=Base frame
        if res_p and res_p[0] == 0:
            pose6 = np.array(res_p[1], dtype=np.float64)
            xyz_m = pose6[:3] / 1000.0
            quat = R.from_euler("xyz", pose6[3:], degrees=True).as_quat()
            self.curr_pose_7d = np.concatenate([xyz_m, quat])
        
        # 2. Joints
        res_q = _maybe_call(self.robot, "GetActualJointPosDegree", 0)
        if res_q and res_q[0] == 0:
            self.curr_q_deg = np.array(res_q[1], dtype=np.float64)

        # 4. Images
        for name, cam in self.cameras.items():
            img = cam.get_frame()
            if img is not None:
                # Optionally crop here if config has IMAGE_CROP
                if name in self.config.IMAGE_CROP:
                    cx, cy, h, w = self.config.IMAGE_CROP[name]
                    img = img[cy:cy+h, cx:cx+w]
                self.curr_images[name] = img

        return self.curr_pose_7d.copy(), self.curr_q_deg.copy(), self.curr_gripper_pos, self.curr_images.copy()

    def _servo_loop(self):
        """High frequency control loop."""
        _maybe_call(self.robot, "ServoMoveStart")
        
        while self.running:
            t0 = time.time()
            
            with self.lock:
                target = self._target_desc_pos.copy() if self._target_desc_pos is not None else None
            
            if target is not None:
                # Send command
                # ServoCart(mode=0, desc_pos=..., pos_gain=..., ...)
                # Note: Robot.py args might vary slightly, ensuring compatibility with standard API
                ret = _maybe_call(
                    self.robot, 
                    "ServoCart",
                    0, # mode
                    target.tolist(),
                    [1.0]*6, # pos_gain
                    0.0, # acc
                    3.0, # vel
                    self.config.CMD_T, # cmdT
                    self.config.FILTER_T, # filterT
                    0.0 # gain
                )
                if ret != 0:
                    # print(f"ServoCart warning: {ret}")
                    pass
            
            dt = time.time() - t0
            sleep_time = max(0.0, self.config.CMD_T - dt)
            time.sleep(sleep_time)

    def joint_reset(self, target_joints_deg):
        """Pause servo, move joints, resume servo."""
        if self.fake_env:
            return

        # Pause servo loop logic (but keep thread alive?)
        # The thread continues but we set target to None or block it?
        # Better to stop servo mode on robot.
        
        # 1. Stop Servo Mode
        with self.lock:
            # We temporarily pause sending commands in the loop?
            # Actually fairino_server.py calls ServoMoveEnd.
            pass
        
        # We need to coordinate with the thread. 
        # Simplest way: Stop thread, move, start thread.
        self.stop(close_cameras=False)
        time.sleep(0.1)

        _maybe_call(self.robot, "ResetAllError")
        
        # 2. MoveJ
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
        
        # Wait for completion (Blocking until target reached)
        t_start = time.time()
        while True:
            res_q = _maybe_call(self.robot, "GetActualJointPosDegree", 0)
            # Handle both tuple (0, data) and int error_code returns
            if res_q is not None:
                if isinstance(res_q, (list, tuple)) and len(res_q) >= 2 and res_q[0] == 0:
                    curr_q = np.array(res_q[1], dtype=np.float64)
                    target_q = np.array(target_joints_deg, dtype=np.float64)
                    # Check if all joints are within tolerance
                    max_error = np.max(np.abs(curr_q - target_q))
                    if max_error < 1.0:  # 1 degree tolerance
                        break
                elif isinstance(res_q, int) and res_q != 0:
                    print(f"Warning: GetActualJointPosDegree returned error code: {res_q}")

            if time.time() - t_start > 15.0:  # 15s timeout
                print(f"Warning: joint_reset timed out. Max error: {max_error if 'max_error' in locals() else 'unknown'}")
                break
            
            time.sleep(0.1)
            
        time.sleep(0.5)

        # 3. Clear targets
        with self.lock:
            self._target_pose_7d = None
            self._target_desc_pos = None

        # 4. Restart
        self.start()


class FairinoEnv(gym.Env):
    """Gym env interacting with Fairino Robot via Controller."""

    def __init__(
        self,
        hz=20,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.fake_env = fake_env
        if config is None:
            config = DefaultEnvConfig()
        self.config = config
        
        self.hz = int(hz)
        self.max_episode_length = int(config.MAX_EPISODE_LENGTH)
        self.gripper_sleep = float(config.GRIPPER_SLEEP)
        self.joint_reset_cycle = int(config.JOINT_RESET_PERIOD)
        
        self.action_scale = np.asarray(config.ACTION_SCALE, dtype=np.float64)
        
        # Action normalization ranges (for [-1, 1] input)
        self.max_pos_delta = float(config.MAX_POS_DELTA)
        self.max_rot_delta = float(config.MAX_ROT_DELTA)
        
        # Initialize Controller
        self.controller = FairinoServoController(config.ROBOT_IP, config, fake_env=fake_env)
        
        # Initial State
        self.resetpos = np.concatenate([config.RESET_POSE[:3], _euler_2_quat(config.RESET_POSE[3:])])
        self.currpos = self.resetpos.copy()
        
        # Start Control Loop
        if not self.fake_env:
            self.controller.start()
            
        self.cycle_count = 0
        self.curr_path_length = 0
        self.last_gripper_act = time.time()
        
        self.action_space = gym.spaces.Box(np.ones((7,), dtype=np.float32) * -1, np.ones((7,), dtype=np.float32))
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
            })
        })

    def step(self, action: np.ndarray) -> tuple:
        start_time = time.time()
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        
        # Denormalize action from [-1, 1] to actual ranges
        # Position: [-1, 1] -> [-max_pos_delta, max_pos_delta]
        xyz_delta_normalized = action[:3]
        xyz_delta = xyz_delta_normalized * self.max_pos_delta
        
        # Rotation: [-1, 1] -> [-max_rot_delta, max_rot_delta]
        rot_delta_normalized = action[3:6]
        rot_delta = rot_delta_normalized * self.max_rot_delta

        # 1. Compute Next Pose based on Denormalized Action
        nextpos = self.currpos.copy()
        nextpos[:3] = nextpos[:3] + xyz_delta

        # Orientation
        nextpos[3:] = (
            R.from_euler("xyz", rot_delta) * R.from_quat(self.currpos[3:])
        ).as_quat()

        # 2. Update Controller Target
        self.controller.set_target_pose(nextpos)
        
        # 3. Handle Gripper
        gripper_action = float(action[6])
        self._handle_gripper(gripper_action)
        
        # 4. Update internal state
        self.currpos = nextpos
        
        self.curr_path_length += 1
        
        # 5. Wait for Gym Hz
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        # 6. Get Observation
        obs = self._get_obs()
        reward = 0 # Implement reward logic if needed
        done = self.curr_path_length >= self.max_episode_length
        
        return obs, int(reward), done, False, {"succeed": False}

    def reset(self, joint_reset=False, **kwargs):
        print(f"DEBUG: FairinoEnv.reset called with joint_reset={joint_reset}, fake_env={self.fake_env}")
        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            joint_reset = True
            
        if self.fake_env:
            self.currpos = self.resetpos.copy()
            return self._get_obs(), {}

        # Perform Joint Reset
        if joint_reset:
            self.controller.joint_reset(self.config.RESET_JOINT_TARGET)
            input(f"Joint reset completed! Reset pose: {self.resetpos}, press Enter to continue...")
        else:
            # If not joint reset, we might want to linearly move to reset pose
            # For now, just set target and wait a bit
            self.controller.set_target_pose(self.resetpos)
            time.sleep(2.0)
        
        input(f"Please replace the object and press Enter to continue...")
        input(f"Are you sure the object is replaced? Press Enter to continue...")

        # Update current pos from actual robot state
        self.currpos, _, _, _ = self.controller.get_state()
        
        self.curr_path_length = 0
        return self._get_obs(), {}

    def _handle_gripper(self, action):
        now = time.time()
        if now - self.last_gripper_act < self.gripper_sleep:
            return
        
        # Placeholder for gripper logic using _maybe_call
        if action <= -0.5:
             # Close
             # _maybe_call(self.controller.robot, "MoveGripper", ...)
             self.last_gripper_act = now
        elif action >= 0.5:
             # Open
             self.last_gripper_act = now

    def _get_obs(self) -> dict:
        # Get latest from controller
        pose_7d, q_deg, g_pos, images = self.controller.get_state()
        
        # If we are controlling tightly, self.currpos should be close to pose_7d.
        # We update self.currpos to actual to prevent drift accumulation?
        # Yes, good practice to re-sync.
        self.currpos = pose_7d
        
        return {
            "state": {
                "tcp_pose": pose_7d,
                # Simulate 2-finger gripper by duplicating position
                "gripper_pose": np.array([g_pos, g_pos]),
                "joint_pose": np.deg2rad(q_deg)
            },
            "images": images
        }

    def close(self):
        self.controller.stop()


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

        self.env = FairinoEnv(
            fake_env=kwargs.get("fake_env", False),
            config=config,
            save_video=kwargs.get("save_video", False)
        )
        self._current_obs = None

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._current_obs = obs
        return self.get_observation(obs), reward, self.is_done(), info

    def reset(self):
        obs, _ = self.env.reset(joint_reset=True)
        self._current_obs = obs
        return self.get_observation(obs)

    def reset_to(self, state):
        return self.reset()

    def render(self, mode="human", height=None, width=None, camera_name=None):
        return None

    def get_observation(self, di=None):
        if di is None:
            di = self._current_obs
            if di is None:
                di = self.env._get_obs()
        
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
                # Ensure _image suffix
                key = k if k.endswith("_image") else f"{k}_image"
                ret[key] = v
                if self.postprocess_visual_obs:
                     ret[key] = ObsUtils.process_obs(obs=ret[key], obs_key=key)

        return ret

    def get_state(self):
        return self.get_observation()

    def get_reward(self):
        return 0

    def get_goal(self):
        return None

    def set_goal(self, **kwargs):
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
