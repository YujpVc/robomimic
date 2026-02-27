import robomimic.envs.env_base as EB
from robomimic.envs.env_base import EnvType
import genesis as gs
import numpy as np
from pyquaternion import Quaternion
import cv2
import time
import os


# 用于将上下文传递给 is_success 函数的辅助类
class ConditionContext:
    def __init__(self, scene, robot, movable_objects):
        self.scene = scene
        self.franka = robot
        self.movable_objects = movable_objects

class GenesisEnvWrapper(EB.EnvBase):
    """
    一个与 robomimic 兼容的 Genesis 环境封装器。
    这个封装器精确地复现了 data_collection.py 脚本中定义的桌面环境，
    确保了状态表示和动作执行的一致性，从而能够正确回放采集的数据。
    """

    def __init__(self, env_name=None, env_config=None, camera_width=84, camera_height=84, **kwargs):
        if env_name is not None:
            if env_config is None:
                env_config = {}
            env_config = env_config.copy()
            env_config["env_name"] = env_name
        if env_config is None:
            env_config = {}

        self.custom_is_success = None
        # 如果 env_config 中有 condition_file，则加载它
        self.condition_file = env_config.get("condition_file", None)
        if self.condition_file and os.path.exists(self.condition_file):
            self._load_condition_file()

        # 调用父类构造函数
        super().__init__(env_config, env_type=EnvType.GENESIS, **kwargs)

        self.reward_shaping = False
        self.post_process_images = False

        # --- Optional safety gating for IK ---
        # Keep these 3 knobs to avoid IK solutions that are invalid or require a big joint jump ("绕一大圈").
        safety_cfg = env_config if isinstance(env_config, dict) else {}
        self.safety_freeze_on_ik_fail = bool(safety_cfg.get("safety_freeze_on_ik_fail", False))
        self.safety_freeze_on_large_joint_jump = bool(safety_cfg.get("safety_freeze_on_large_joint_jump", True))
        self.safety_max_abs_joint_delta = float(safety_cfg.get("safety_max_abs_joint_delta", 0.3))  # rad

        # 与数据采集脚本一致的机器人控制参数
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        # 夹爪控制参数 - 不再需要上次夹爪信号
        self.GRASP_FORCE = 10.0  # 夹持力常数

        # 1. 初始化 Genesis 场景 (与采集脚本完全一致)
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, 1.0, 1.8),
                camera_lookat=(0.65, 1.0, 1.0),
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
            show_viewer=False,  # 通常在训练/回放时关闭默认查看器
            show_FPS=False
        )
        self.env = self.scene
        self.simulator = self.env

        # 2. 添加所有场景实体 (与采集脚本完全一致)
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml",
                           pos=(0.0, 1.0, 0.88))
        )
        self.desk_scene = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/home/yujp/Genesis/my_models/scenes/libero_study_base_style.xml",
                pos=(0.5, 1.0, 0.0), scale=1.0,
            )
        )
        self.shelf = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/home/yujp/Genesis/my_models/turbosquid_objects/wooden_shelf/wooden_shelf.xml",
                pos=(0.6, 0.65, 0.88), quat=(0, 0, 0, 1), scale=1,
            )
        )
        self.tray = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/home/yujp/Genesis/my_models/turbosquid_objects/wooden_tray/wooden_tray.xml",
                pos=(0.4, 1.3, 0.88), scale=1
            )
        )
        self.box = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/home/yujp/Genesis/my_models/turbosquid_objects/white_storage_box/white_storage_box.xml",
                pos=(0.35, 0.8, 0.88), scale=1,
            )
        )
        # 三个可移动方块（位置保持原来的，不加缩放 / 旋转）
        # 颜色分别为：红 / 蓝 / 黄
        self.object_1 = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.7, 0.85, 0.88),
            ),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
        )
        self.object_2 = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.75, 1.15, 1.0),
            ),
            surface=gs.surfaces.Default(color=(0.0, 0.0, 1.0)),
        )
        self.object_3 = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.75, 1.0, 0.88),
            ),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 0.0)),
        )

        # 3. 设置摄像头 (使用 robomimic 命名约定)
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.cameras = {}

        self.cameras["agentview_image"] = self.scene.add_camera(
            res=(self.camera_width, self.camera_height),
            pos=(2.5, 1.0, 1.8), lookat=(0.65, 1.0, 1.0), fov=30, GUI=False
        )
        self.cameras["show_image"] = self.scene.add_camera(
            res=(self.camera_width, self.camera_height),
            pos=(0, 0, 0), lookat=(0, 0, 0), fov=60, GUI=False
        )
        self.cameras["robot0_eye_in_hand_image"] = self.scene.add_camera(
            res=(self.camera_width, self.camera_height),
            pos=(0, 0, 0), lookat=(0, 0, 0), fov=60, GUI=False
        )
        # 用于高质量渲染的摄像头
        self.cameras["render_view"] = self.scene.add_camera(
            res=(512, 512),
            pos=(2.5, 1.0, 1.8), lookat=(0.65, 1.0, 1.0), fov=30, GUI=False
        )

        # 构建场景
        self.scene.build()

        # 4. 初始化依赖场景构建的组件
        self.end_effector = self.robot.get_link("hand")
        self.robot.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
        self.robot.set_dofs_kv([450, 450, 350, 350, 200, 200, 200, 10, 10])
        self.robot.set_dofs_force_range(
            [-87, -87, -87, -87, -12, -12, -12, -100, -100],
            [87, 87, 87, 87, 12, 12, 12, 100, 100]
        )

        # 5. 管理可移动物体 (与采集脚本一致)
        # MovableObject 辅助类
        class MovableObject:
            def __init__(self, entity, initial_pos, initial_quat):
                self.entity = entity
                self.initial_pos = np.array(initial_pos)
                self.initial_quat = np.array(initial_quat)

            def get_state(self):
                pos = self.entity.get_pos().cpu().numpy().squeeze()
                quat = self.entity.get_quat().cpu().numpy().squeeze()
                vel = self.entity.get_vel().cpu().numpy().squeeze()
                ang = self.entity.get_ang().cpu().numpy().squeeze()
                return pos, quat, vel, ang

        self.movable_objects = [
            MovableObject(self.object_1, [0.70, 0.85, 0.90], [0, 0, 0.707, 0.707]),
            MovableObject(self.object_2, [0.70, 1.15, 0.90], [0, 0, 0.707, 0.707]),
            MovableObject(self.object_3, [0.70, 1.0, 0.90], [0, 0, 0.707, 0.707])
        ]

        # 6. 初始化控制状态
        self.init_qpos = np.array([-2.0988, -1.4417,  1.5711, -1.7141,  1.4430,  1.5895,  0.1152,  0.04, 0.04])
        # 将在 reset() 中初始化
        self.current_pos = None
        self.current_quat = None

    @classmethod
    def create_for_data_processing(cls, env_name, camera_height, camera_width, reward_shaping, **kwargs):
        env_config = {
            "env_name": env_name if env_name else "Genesis Franka Environment"
        }
        env = cls(
            env_config=env_config,
            camera_width=camera_width,
            camera_height=camera_height
        )
        # 初始化观测规范
        from robomimic.utils.obs_utils import initialize_obs_utils_with_obs_specs
        initialize_obs_utils_with_obs_specs(
            obs_modality_specs={
                "obs": {
                    "low_dim": [
                        "robot0_joint_pos", "robot0_joint_vel", "robot0_joint_pos_cos",
                        "robot0_joint_pos_sin", "robot0_gripper_qpos", "robot0_gripper_qvel",
                        "robot0_eef_pos", "robot0_eef_quat", "robot0_eef_vel_lin",
                        "robot0_eef_vel_ang", "object"
                    ],
                    "rgb": ["agentview_image", "robot0_eye_in_hand_image"]
                }
            }
        )
        env.reward_shaping = reward_shaping
        return env

    def step(self, action):
        """
        根据采集的动作指令，在仿真环境中执行一步。
        action: 包含位置、旋转和夹爪控制的7维向量。
                - action[0:3]: delta_pos (世界坐标系下的位置增量)
                - action[3:6]: delta_rot (世界坐标系下的旋转矢量)
                - action[6]:   gripper_signal (夹爪信号, >0 表示闭合, <=0 表示张开)
        """
        # 1. 解析动作
        # NOTE: 缩放后的数据集中位置动作放大了,此处需还原
        delta_pos = np.clip(action[:3], -1.0, 1.0)
        delta_pos = action[:3] / 400.0
        delta_rot_vec = action[3:6] / 50.0
        gripper_signal = action[6]

        # 2. 计算本步期望的目标位姿（先不立刻写回 self.current_*）
        proposed_pos = self.current_pos + delta_pos
        proposed_quat = self.current_quat

        # 3. 计算本步期望的目标姿态（世界坐标系增量，前乘）
        angle = np.linalg.norm(delta_rot_vec)
        if angle > 1e-6:
            axis = delta_rot_vec / angle
            q_inc = Quaternion(axis=axis, angle=angle)  # 增量旋转
            q_current = Quaternion(self.current_quat)   # 当前姿态
            q_new = q_inc * q_current
            proposed_quat = q_new.normalised.elements

        # 4. 应用逆运动学 (IK) + 可选安全检查
        freeze_this_step = False
        qpos = None
        qpos_cpu = None
        try:
            qpos = self.robot.inverse_kinematics(
                link=self.end_effector, pos=proposed_pos, quat=proposed_quat
            )
            # convert to CPU numpy for checks (IK may return torch tensor)
            if hasattr(qpos, "detach"):
                qpos_cpu = qpos.detach().cpu().numpy().reshape(-1)
            else:
                qpos_cpu = np.asarray(qpos).reshape(-1)

            if (not np.all(np.isfinite(qpos_cpu))) and self.safety_freeze_on_ik_fail:
                freeze_this_step = True

            if (not freeze_this_step) and self.safety_freeze_on_large_joint_jump:
                cur_q = self.robot.get_dofs_position().cpu().numpy().reshape(-1)[:7]
                tgt_q = qpos_cpu[:7]
                if float(np.max(np.abs(tgt_q - cur_q))) > self.safety_max_abs_joint_delta:
                    freeze_this_step = True
        except Exception:
            if self.safety_freeze_on_ik_fail:
                freeze_this_step = True

        if not freeze_this_step:
            # only commit targets if we accept this step
            self.current_pos = proposed_pos
            self.current_quat = proposed_quat
            self.robot.control_dofs_position(qpos[:-2], self.motors_dof)

        # 5. 控制夹爪（使用位置控制）
        if gripper_signal > 0:  # 闭合夹爪
            # 位控闭合并持续保持闭合
            self.robot.control_dofs_position([0.0, 0.0], self.fingers_dof)
        else:  # 张开夹爪
            self.robot.control_dofs_position([0.04, 0.04], self.fingers_dof)

        # 6. 推进仿真以匹配控制频率 (20Hz)
        control_freq = 20
        action_duration = 1.0 / control_freq
        sim_dt = self.scene.sim_options.dt
        num_sim_steps = int(round(action_duration / sim_dt))
        for _ in range(num_sim_steps):
            self.scene.step()

        # 7. 获取结果
        obs = self.get_observation()
        reward = self.get_reward()
        done = self.is_done()["task"]
        return obs, reward, done, {}

    def reset(self):
        """重置环境到初始状态，包括对可移动物体位置的随机化。"""
        self.simulator.reset()

        # 重置机器人目标位姿 (在固定初始关节配置附近加入小范围随机扰动)
        # 仅对前 7 个关节（臂部）加入随机偏移，夹爪保持不变
        noisy_qpos = self.init_qpos.copy()
        arm_noise = np.random.uniform(-0.05, 0.05, size=7)  # ~±3 度
        noisy_qpos[:7] += arm_noise
        self.robot.set_dofs_position(noisy_qpos)

        # 重置所有可移动物体位置（带随机偏移）
        for obj in self.movable_objects:
            rand_offset = np.random.uniform(-0.01, 0.01, 2)
            reset_pos = obj.initial_pos.copy()
            reset_pos[0] += rand_offset[0]
            reset_pos[1] += rand_offset[1]
            obj.entity.set_pos(reset_pos)
            # 随机旋转（默认只绕 Z 轴随机 yaw，避免把物体“翻倒”）
            rand_yaw = np.random.uniform(-np.pi, np.pi)
            q_base = Quaternion(obj.initial_quat.copy())
            q_yaw = Quaternion(axis=[0, 0, 1], angle=rand_yaw)
            q_new = (q_yaw * q_base).normalised
            obj.entity.set_quat(q_new.elements)

        self.scene.step()

        # 更新IK控制器的目标位姿以匹配重置后的状态
        self.current_pos = self.end_effector.get_pos().cpu().numpy().squeeze()
        self.current_quat = self.end_effector.get_quat().cpu().numpy().squeeze()

        print("----------------reset success----------------")

        return self.get_observation()

    def reset_to(self, state):
        """
        根据给定的状态向量重置环境。
        这是确保回放正确的关键函数。
        """
        self.simulator.reset()

        full_state_vector = state["states"]

        # 解析状态向量 (必须与 get_state 的结构完全一致)
        current_idx = 1  # 跳过时间戳

        # 设置机器人状态
        robot_qpos = full_state_vector[current_idx: current_idx + 9]
        current_idx += 9
        robot_qvel = full_state_vector[current_idx: current_idx + 9]
        current_idx += 9
        self.robot.set_dofs_position(robot_qpos)
        self.robot.set_dofs_velocity(robot_qvel)

        # 设置所有可移动物体的状态
        for obj in self.movable_objects:
            pos = full_state_vector[current_idx: current_idx + 3]
            current_idx += 3
            quat = full_state_vector[current_idx: current_idx + 4]
            current_idx += 4
            vel = full_state_vector[current_idx: current_idx + 3]
            current_idx += 3
            ang = full_state_vector[current_idx: current_idx + 3]
            current_idx += 3
            obj.entity.set_pos(pos)
            obj.entity.set_quat(quat)

        # 更新IK控制器的目标位姿以匹配重置后的状态
        self.current_pos = self.end_effector.get_pos().cpu().numpy().squeeze()
        self.current_quat = self.end_effector.get_quat().cpu().numpy().squeeze()

        self.scene.step()
        return self.get_observation()

    def get_state(self):
        """
        获取当前环境的完整状态向量。
        其结构必须与数据采集脚本中保存到HDF5文件的结构完全一致。
        """
        # 获取机器人状态
        robot_qpos = self.robot.get_dofs_position().cpu().numpy()
        robot_qvel = self.robot.get_dofs_velocity().cpu().numpy()

        # 收集所有可移动物体的状态
        obj_states_flat = []
        for obj in self.movable_objects:
            pos, quat, vel, ang = obj.get_state()
            obj_states_flat.extend(list(pos.flatten()))
            obj_states_flat.extend(list(quat.flatten()))
            obj_states_flat.extend(list(vel.flatten()))
            obj_states_flat.extend(list(ang.flatten()))

        # 按照采集脚本的格式拼接
        # 格式: [time(1), robot_qpos(9), robot_qvel(9), all_obj_states(N*13)]
        full_state = np.concatenate(
            [[time.time()], robot_qpos, robot_qvel, np.array(obj_states_flat)]
        )
        return {"states": full_state}

    def get_observation(self):
        """获取 robomimic 格式的观测数据。"""
        joint_pos = self.robot.get_dofs_position().cpu().numpy().copy()
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy().copy()
        eef_pos = self.end_effector.get_pos().cpu().numpy().squeeze().copy()
        eef_quat = self.end_effector.get_quat().cpu().numpy().squeeze().copy()

        # 获取并拼接所有可移动物体的状态作为 "object" 观测
        object_states = []
        for obj in self.movable_objects:
            pos, quat, _, _ = obj.get_state()
            object_states.extend(pos)
            object_states.extend(quat)

        obs = {
            "robot0_joint_pos": joint_pos[:7],
            "robot0_joint_vel": joint_vel[:7],
            "robot0_joint_pos_cos": np.cos(joint_pos[:7]),
            "robot0_joint_pos_sin": np.sin(joint_pos[:7]),
            "robot0_gripper_qpos": joint_pos[7:],
            "robot0_gripper_qvel": joint_vel[7:],
            "robot0_eef_pos": eef_pos,
            "robot0_eef_quat": eef_quat,
            "object": np.array(object_states),
        }

        # 更新摄像头位置并获取图像
        current_q = Quaternion(eef_quat)
        rot_z = Quaternion(axis=[0, 0, 1], angle=-np.pi / 2)
        adjusted_q = current_q * rot_z
        x_dir = np.array(adjusted_q.rotate([1, 0, 0]))
        y_dir = np.array(adjusted_q.rotate([0, 1, 0]))
        z_dir = np.array(adjusted_q.rotate([0, 0, 1]))

        # agentview 是固定的，无需更新
        img_agent = self.cameras["agentview_image"].render(rgb=True)[0].copy()
        obs["agentview_image"] = img_agent.transpose(2, 0, 1) if self.post_process_images else img_agent

        # 更新手眼摄像头
        gripper_cam_pos = eef_pos + y_dir * 0.075 - z_dir * 0.05
        gripper_cam_lookat = gripper_cam_pos + z_dir * 0.5
        self.cameras["show_image"].set_pose(
            pos=gripper_cam_pos.astype(np.float32),
            lookat=gripper_cam_lookat.astype(np.float32),
            up=y_dir.astype(np.float32)
        )
        img_show = self.cameras["show_image"].render(rgb=True)[0].copy()
        obs["show_image"] = img_show.transpose(2, 0, 1) if self.post_process_images else img_show

        # 更新手腕摄像头
        wrist_cam_pos = eef_pos
        wrist_cam_lookat = wrist_cam_pos + z_dir * 0.05
        self.cameras["robot0_eye_in_hand_image"].set_pose(
            pos=wrist_cam_pos.astype(np.float32),
            lookat=wrist_cam_lookat.astype(np.float32),
            up=y_dir.astype(np.float32)
        )
        img_hand = self.cameras["robot0_eye_in_hand_image"].render(rgb=True)[0].copy()
        obs["robot0_eye_in_hand_image"] = img_hand.transpose(2, 0, 1) if self.post_process_images else img_hand

        return obs

    def _load_condition_file(self):
        """从外部文件加载 is_success 函数。"""
        try:
            with open(self.condition_file, 'r') as f:
                condition_code = f.read()
            local_vars = {}
            # 提供一个默认的 ConditionContext
            context = ConditionContext(self.scene, self.robot, self.movable_objects)
            exec(condition_code, globals(), local_vars)
            if 'is_success' in local_vars:
                # 绑定self和上下文到函数
                self.custom_is_success = lambda: local_vars['is_success'](context)
                print(f"成功加载条件文件: {self.condition_file}")
            else:
                raise ValueError("条件文件中未定义 is_success 函数")
        except Exception as e:
            print(f"加载条件文件失败: {str(e)}")
            self.custom_is_success = None

    def is_success(self):
        """检查任务是否成功。"""
        if self.custom_is_success:
            result = self.custom_is_success()
            # 确保返回的是字典格式
            return result if isinstance(result, dict) else {"task": bool(result)}
        return {"task": False}  # 默认返回

    def render(self, mode="human", camera_name="agentview_image", waitkey=1, **kwargs):
        if mode == "collect":
            # 获取三个摄像头的图像
            agent_image = self.cameras["agentview_image"].render(rgb=True)[0].copy()
            eye_image = self.cameras["show_image"].render(rgb=True)[0].copy()
            gripper_image = self.cameras["robot0_eye_in_hand_image"].render(rgb=True)[0].copy()

            # 转换颜色空间并水平拼接
            agent_bgr = cv2.cvtColor(agent_image, cv2.COLOR_RGB2BGR)
            eye_bgr = cv2.cvtColor(eye_image, cv2.COLOR_RGB2BGR)
            gripper_bgr = cv2.cvtColor(gripper_image, cv2.COLOR_RGB2BGR)
            combined = np.hstack([agent_bgr, eye_bgr, gripper_bgr])

            # 显示图像
            cv2.imshow('Collect View - [Agent | Eye-in-Hand | Gripper]', combined)
            cv2.waitKey(waitkey)
            return combined
        else:
            # 检查摄像头名称是否存在，如果不存在则回退到agentview
            if camera_name not in self.cameras:
                print(f"Warning: Camera '{camera_name}' not found. Defaulting to 'agentview_image'.")
                camera_name = "agentview_image"

            cam = self.cameras[camera_name]
            rgb = cam.render(rgb=True)[0].copy()

            if mode == "human":
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow(camera_name, bgr)
                cv2.waitKey(waitkey)
                return None
            elif mode == "rgb_array":
                return rgb
            return None

    # --- robomimic boilerplate ---
    @property
    def action_dimension(self):
        return 7  # [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]

    @property
    def name(self):
        return "Genesis_Franka_Environment"

    @property
    def type(self):
        return EnvType.GENESIS

    def get_reward(self):
        return float(self.is_success()["task"])

    def is_done(self):
        return self.is_success()

    def serialize(self):
        return {
            "type": EnvType.GENESIS,
            "env_name": self.name,
            "env_kwargs": {}
        }

    @property
    def rollout_exceptions(self):
        return (Exception,)

    def get_goal(self):
        return self.get_observation()

    def set_goal(self, **kwargs):
        pass