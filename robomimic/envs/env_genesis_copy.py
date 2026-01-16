import robomimic.envs.env_base as EB
from robomimic.envs.env_base import EnvType
import genesis as gs
import numpy as np
from pyquaternion import Quaternion
import cv2
import time

class GenesisEnvWrapper(EB.EnvBase):
    def __init__(self, env_name=None, env_config=None, camera_width=84, camera_height=84, **kwargs):
        if env_name is not None:
            if env_config is None:
                env_config = {}
            env_config = env_config.copy()
            env_config["env_name"] = env_name
        if env_config is None:
            env_config = {}

        self.custom_is_success = None

        # 调用父类构造函数
        super().__init__(env_config, env_type=EnvType.GENESIS, **kwargs)

        self.reward_shaping = False
        self.post_process_images = False
        # 添加时间控制变量
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)

        self.last_gripper_signal = -1.0

        # 初始化 Genesis 场景
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
            show_viewer=False,
            show_FPS=False
        )

        self.env = self.scene
        self.simulator = self.env
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="/home/yujp/Genesis/genesis/assets/xml/franka_emika_panda/panda.xml")
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))
        self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="/home/yujp/Genesis/my_models/wooden_two_layer_shelf/wooden_two_layer_shelf.obj",
                scale=0.1,
                pos=(0.0, 0.75, 0.0),
            )
        )

        self.scene.add_entity(
            gs.morphs.Box(size=(0.06, 0.06, 0.02), pos=(0.5, 0.0, 0.01)),
            surface=gs.surfaces.Default(color=(1.0, 0, 0, 1.0))
        )

        # 添加摄像头
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.cameras = {}

        # 全局视角摄像头（固定）
        self.cameras["agentview"] = self.scene.add_camera(
            res=(self.camera_width, self.camera_height),
            pos=(2.5, 0, 1.2),
            lookat=(0.65, 0.0, 0.25),
            fov=30,
            GUI=False
        )

        # 腕部视角摄像头（动态调整位置）
        self.cameras["gripper_view"] = self.scene.add_camera(
            res=(self.camera_width, self.camera_height),
            pos=(0, 0, 0),  # 位置后续动态更新
            lookat=(0, 0, 0),
            fov=60,
            GUI=False
        )

        # 末端执行器视角摄像头（动态调整位置） - 名称与观测键统一
        self.cameras["robot0_eye_in_hand_image"] = self.scene.add_camera(
            res=(self.camera_width, self.camera_height),
            pos=(0, 0, 0),  # 位置后续动态更新
            lookat=(0, 0, 0),
            fov=60,
            GUI=False
        )

        # 高分辨率渲染摄像头（可选）
        self.cameras["render_view"] = self.scene.add_camera(
            res=(512, 512),
            pos=(2.5, 0, 1.2),
            lookat=(0.65, 0.0, 0.25),
            fov=30,
            GUI=False
        )

        # 构建场景
        self.scene.build()

        # 初始化依赖场景构建的组件
        self.end_effector = self.robot.get_link("hand")
        self.robot.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
        self.robot.set_dofs_kv([450, 450, 350, 350, 200, 200, 200, 10, 10])
        self.robot.set_dofs_force_range(
            [-87, -87, -87, -87, -12, -12, -12, -100, -100],
            [87, 87, 87, 87, 12, 12, 12, 100, 100]
        )

        # 初始目标位置和姿态
        self.current_pos = np.array([0.65, 0.0, 0.3])
        self.current_quat = np.array([0, 1, 0, 0])  # [w, x, y, z]

        # 新增与数据收集一致的常量
        self.SPEED_XY = 0.5      # 平面移动速度（米/秒）
        self.SPEED_Z = 0.2       # 垂直移动速度（米/秒）
        self.ROT_SPEED = 1.0     # 旋转速度（弧度/秒）
        self.DATA_INTERVAL = 0.05  # 20Hz数据记录间隔

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

    # def step(self, action):
    #     delta_pos = action[:3]
    #     delta_rot_vec = action[3:6]
    #     gripper_signal = action[6]
    #
    #     # 1. 位置更新
    #     self.current_pos += delta_pos
    #
    #     # 2. 姿态更新
    #     angle = np.linalg.norm(delta_rot_vec)
    #     if angle > 1e-6:  # 避免除以零
    #         axis = delta_rot_vec / angle
    #         q_inc = Quaternion(axis=axis, angle=angle)  # 从数据中读取的“世界坐标系”旋转 (The world-frame rotation from data)
    #         q_current = Quaternion(self.current_quat)  # 上一步的姿态 (Orientation from previous step)
    #
    #         # --- 核心修改 / CORE FIX ---
    #         # 您的数据动作用 q_current * q_last.inverse 计算，这是一个世界坐标系下的旋转。
    #         # 因此，在应用它时，需要使用前乘 (pre-multiplication) 来更新姿态。
    #         # Your data action was calculated with q_current * q_last.inverse, which is a rotation in the WORLD frame.
    #         # Therefore, to apply it, you must use pre-multiplication.
    #         #
    #         # 原代码 (Original Code - applies rotation in LOCAL frame):
    #         # q_new = q_current * q_inc
    #         #
    #         # 修改后代码 (Corrected Code - applies rotation in WORLD frame):
    #         q_new = q_inc * q_current
    #
    #         self.current_quat = q_new.normalised.elements
    #
    #     # 通过逆运动学计算新的关节角度
    #     qpos = self.robot.inverse_kinematics(link=self.end_effector, pos=self.current_pos, quat=self.current_quat)
    #
    #     # 控制机械臂前7个关节和夹爪
    #     self.robot.control_dofs_position(qpos[:-2], self.motors_dof)
    #     gripper_target = 0.0 if gripper_signal > 0 else 0.04
    #     self.robot.control_dofs_position([gripper_target, gripper_target], self.fingers_dof)
    #
    #     # 根据控制频率推进仿真 (保持时间同步)
    #     control_freq = 20
    #     action_duration = 1.0 / control_freq
    #     sim_dt = self.scene.sim_options.dt
    #     num_sim_steps = int(round(action_duration / sim_dt))
    #
    #     for _ in range(num_sim_steps):
    #         self.scene.step()
    #
    #     # 获取观测信息
    #     obs = self.get_observation()
    #     reward = self.get_reward()
    #     done = self.is_done()
    #
    #     return obs, reward, done, {}

    def step(self, action):
        """
        根据采集的动作指令，在仿真环境中执行一步。
        action: 包含位置、旋转和夹爪控制的7维向量。
                - action[0:3]: delta_pos (世界坐标系下的位置增量)
                - action[3:6]: delta_rot_vec (世界坐标系下的旋转矢量)
                - action[6]:   gripper_signal (夹爪信号, >0 表示闭合)
        """
        # 1. 解析动作
        delta_pos = action[:3]
        delta_rot_vec = action[3:6]
        gripper_signal = action[6]

        # 2. 更新目标位置 (直接在世界坐标系下累加，这是正确的)
        self.current_pos += delta_pos

        # 3. 更新目标姿态
        angle = np.linalg.norm(delta_rot_vec)
        # 只有在旋转角度不为零时才更新姿态，避免计算错误
        if angle > 1e-6:
            axis = delta_rot_vec / angle
            # 将旋转矢量转换为增量四元数 q_inc
            q_inc = Quaternion(axis=axis, angle=angle)
            # 获取当前姿态的四元数
            q_current = Quaternion(self.current_quat)

            # --- 核心修正 (THE CORE FIX) ---
            # 采集的动作是在世界坐标系下的旋转增量，因此必须使用前乘 (pre-multiplication)
            # q_new = 世界旋转增量 * 当前世界姿态
            q_new = q_inc * q_current

            # 更新并归一化姿态
            self.current_quat = q_new.normalised.elements

        # 4. 应用逆运动学 (IK)
        # 使用更新后的目标位姿，计算机器人各关节所需的目标角度
        qpos = self.robot.inverse_kinematics(
            link=self.end_effector,
            pos=self.current_pos,
            quat=self.current_quat
        )

        # 5. 控制机器人
        # 控制机械臂主臂（前7个自由度）
        self.robot.control_dofs_position(qpos[:-2], self.motors_dof)

        # 控制夹爪 (根据信号判断目标开合度)
        # 优化：仅当夹爪信号从上一步到这一步发生变化时，才发送新指令，避免重复
        if gripper_signal != self.last_gripper_signal:
            # gripper_signal > 0 代表闭合指令
            if gripper_signal > 0:
                gripper_target = 0.0  # 闭合位置
            # gripper_signal <= 0 代表张开指令
            else:
                gripper_target = 0.04  # 张开位置

            # 发送控制指令
            self.robot.control_dofs_position([gripper_target, gripper_target], self.fingers_dof)

            # 更新最后一次的信号状态，为下一次比较做准备
            self.last_gripper_signal = gripper_signal

        # 6. 推进仿真
        # 为了与数据采集频率 (20Hz) 保持同步，每个action执行固定的仿真步数
        control_freq = 20  # 20Hz
        action_duration = 1.0 / control_freq
        sim_dt = self.scene.sim_options.dt
        num_sim_steps = int(round(action_duration / sim_dt))

        for _ in range(num_sim_steps):
            self.scene.step()

        # 7. 获取结果
        obs = self.get_observation()
        reward = self.get_reward()
        done = self.is_done()

        return obs, reward, done, {}

    def reset(self):
        # 重置仿真环境
        self.simulator.reset()

        self.last_gripper_signal = -1.0

        # 重置末端执行器目标状态到初始值
        self.current_pos = np.array([0.65, 0.0, 0.3])
        self.current_quat = np.array([0, 1, 0, 0])

        # 使用逆运动学计算初始位置的关节角度
        qpos = self.robot.inverse_kinematics(
            link=self.end_effector,
            pos=self.current_pos,
            quat=self.current_quat
        )

        # 设置机械臂关节角度
        self.robot.set_dofs_position(qpos)
        self.scene.step()

        # 重置方块位置 - 添加随机化
        cube_entity = self.scene.entities[2]
        target = self.scene.entities[4]

        # 添加随机位置设置
        cube_x = np.random.uniform(0.6, 0.7)  # 在X轴上随机位置
        cube_y = np.random.uniform(-0.05, 0.05)  # 在Y轴上随机位置
        cube_z = 0.02  # Z轴固定高度

        cube_entity.set_pos(np.array([cube_x, cube_y, cube_z]))
        cube_entity.set_quat(np.array([1, 0, 0, 0]))

        target.set_pos(np.array([0.5, 0.0, 0.01]))
        target.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

        self.scene.step()

        print(f"重置方块位置: ({cube_x:.2f}, {cube_y:.2f}, {cube_z:.2f})")

        # 重置计时器
        self.last_step_time = time.time()

        return self.get_observation()

    def reset_to(self, state):
        self.simulator.reset()
        states = state["states"]
        qpos = states[1:17]
        robot_joint_pos = qpos[:9]
        cube_pos = qpos[9:12]
        cube_quat = qpos[12:16]

        # 设置机器人和立方体状态
        self.robot.set_dofs_position(robot_joint_pos)
        cube_entity = self.scene.entities[2]
        cube_entity.set_pos(cube_pos)
        cube_entity.set_quat(cube_quat)

        # 更新末端执行器目标状态
        self.current_pos = self.end_effector.get_pos().cpu().numpy().squeeze()
        self.current_quat = self.end_effector.get_quat().cpu().numpy().squeeze()

        self.last_step_time = time.time()
        return self.get_observation()

    def get_observation(self):
        joint_pos = self.robot.get_dofs_position().cpu().numpy().copy()
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy().copy()
        eef_pos = self.end_effector.get_pos().cpu().numpy().squeeze().copy()
        eef_quat = self.end_effector.get_quat().cpu().numpy().squeeze().copy()
        cube_entity = self.scene.entities[2]

        obs = {
            "robot0_joint_pos": joint_pos[:7],
            "robot0_joint_vel": joint_vel[:7],
            "robot0_joint_pos_cos": np.cos(joint_pos[:7]),
            "robot0_joint_pos_sin": np.sin(joint_pos[:7]),
            "robot0_gripper_qpos": joint_pos[7:],
            "robot0_gripper_qvel": joint_vel[7:],
            "robot0_eef_pos": eef_pos,
            "robot0_eef_quat": eef_quat,
            "robot0_eef_vel_lin": self.end_effector.get_vel().cpu().numpy().squeeze().copy(),
            "robot0_eef_vel_ang": self.end_effector.get_ang().cpu().numpy().squeeze().copy(),
            "object": np.concatenate([
                cube_entity.get_pos().cpu().numpy().squeeze().copy(),
                cube_entity.get_quat().cpu().numpy().squeeze().copy()
            ]),
        }

        # 更新摄像头位置并获取图像
        current_q = Quaternion(eef_quat)
        rot_z = Quaternion(axis=[0, 0, 1], angle=-np.pi / 2)
        adjusted_q = current_q * rot_z
        z_dir = np.array(adjusted_q.rotate([0, 0, 1]))
        y_dir = np.array(adjusted_q.rotate([0, 1, 0]))

        # Agentview
        img_agent = self.cameras["agentview"].render(rgb=True)[0].copy()
        obs["agentview_image"] = img_agent.transpose(2, 0, 1) if self.post_process_images else img_agent

        # Gripper view
        cam_gripper_pos = eef_pos
        cam_gripper_lookat = cam_gripper_pos + z_dir * 0.5
        self.cameras["gripper_view"].set_pose(
            pos=cam_gripper_pos.astype(np.float32),
            lookat=cam_gripper_lookat.astype(np.float32),
            up=y_dir.astype(np.float32)
        )

        # Eye-in-hand view
        cam_in_hand_pos = eef_pos + y_dir * 0.05
        cam_in_hand_lookat = cam_in_hand_pos + z_dir * 0.5
        self.cameras["robot0_eye_in_hand_image"].set_pose(
            pos=cam_in_hand_pos.astype(np.float32),
            lookat=cam_in_hand_lookat.astype(np.float32),
            up=y_dir.astype(np.float32)
        )
        img_hand = self.cameras["robot0_eye_in_hand_image"].render(rgb=True)[0].copy()
        obs["robot0_eye_in_hand_image"] = img_hand.transpose(2, 0, 1) if self.post_process_images else img_hand

        return obs

    def _load_condition_file(self):
        try:
            with open(self.condition_file, 'r') as f:
                condition_code = f.read()
            local_vars = {}
            exec(condition_code, globals(), local_vars)
            if 'is_success' in local_vars:
                self.custom_is_success = local_vars['is_success']
            else:
                raise ValueError("condition_file 中未定义 is_success 函数")
        except Exception as e:
            print(f"加载 condition_file 失败: {str(e)}")
            self.custom_is_success = None

    def is_success(self):
        if self.custom_is_success is not None:
            return self.custom_is_success.__get__(self, GenesisEnvWrapper)()
        return {"task": False}

    def get_reward(self):
        return float(self.is_success()["task"])

    def is_done(self):
        return self.is_success()["task"]

    @property
    def action_dimension(self):
        return 7

    @property
    def name(self):
        return "Genesis_Franka_Environment"

    @property
    def type(self):
        return EnvType.GENESIS

    def serialize(self):
        return {
            "type": EnvType.GENESIS,
            "env_name": "Genesis_Franka_Environment",
            "env_kwargs": {}
        }

    def get_state(self):
        # 状态向量结构必须与数据收集中完全一致
        # [time(1), robot_qpos(9), cube_pos(3), cube_quat(4), robot_qvel(9), cube_vel(3), cube_ang(3)]
        # Total: 1 + 9 + 3 + 4 + 9 + 3 + 3 = 32
        robot_qpos = self.robot.get_dofs_position().cpu().numpy()
        cube_entity = self.scene.entities[2]
        cube_pos = cube_entity.get_pos().cpu().numpy().squeeze()
        cube_quat = cube_entity.get_quat().cpu().numpy().squeeze()

        robot_qvel = self.robot.get_dofs_velocity().cpu().numpy()
        cube_vel = cube_entity.get_vel().cpu().numpy().squeeze()
        cube_ang = cube_entity.get_ang().cpu().numpy().squeeze()

        qpos = np.concatenate([robot_qpos, cube_pos, cube_quat])
        qvel = np.concatenate([robot_qvel, cube_vel, cube_ang])

        full_state = np.concatenate([[time.time()], qpos, qvel])

        return {"states": full_state}

    def get_goal(self):
        return self.get_observation()

    def set_goal(self, **kwargs):
        pass

    @property
    def rollout_exceptions(self):
        return (Exception,)

    def render(self, mode="human", camera_name="agentview", waitkey=1, **kwargs):
        if mode == "collect":
            # 获取三个摄像头的图像
            agent_image = self.cameras["agentview"].render(rgb=True)[0].copy()
            eye_image = self.cameras["robot0_eye_in_hand_image"].render(rgb=True)[0].copy()
            gripper_image = self.cameras["gripper_view"].render(rgb=True)[0].copy()

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
                print(f"Warning: Camera '{camera_name}' not found. Defaulting to 'agentview'.")
                camera_name = "agentview"

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
