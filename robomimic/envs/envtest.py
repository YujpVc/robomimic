import h5py
import numpy as np
import cv2
import time
from env_genesis import GenesisEnvWrapper  # 假设你的环境类保存在这个文件中

def test_hdf5_demo():
    # 初始化环境和视频写入器
    env = GenesisEnvWrapper(
        camera_width=640,
        camera_height=480,
        post_process_images=True
    )

    with h5py.File("/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/demo202503180417/demos.hdf5", "r") as f:
        demos_group = f["data"]

        # 遍历所有demo
        for demo_name in demos_group:
            print(f"Processing {demo_name}...")
            demo = demos_group[demo_name]

            # 获取动作序列
            actions = demo["actions"][:]
            states = demo["states"][:]

            # 重置环境到初始状态
            initial_state = {
                "states": states[0, :]  # 假设第一个状态是初始状态
            }
            obs = env.reset_to(initial_state)

            for i, action in enumerate(actions):
                cycle_start = time.time()  # 记录循环开始时间

                # 执行动作
                obs, reward, done, info = env.step(action)

                # 渲染图像（将waitkey设置为1毫秒确保画面刷新）
                env.render()

                # 计算耗时并保持20Hz频率
                elapsed = time.time() - cycle_start
                sleep_time = 0.05 - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"警告：第{i}步延迟{-sleep_time:.3f}秒")

                if done:
                    print(f"Episode terminated early at step {i}")
                    break

if __name__ == "__main__":
    test_hdf5_demo()