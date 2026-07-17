import argparse
from pathlib import Path

import gym
import numpy as np
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "scripts" / "config.yml"
IMAGE_SHAPE = (50, 50, 3)


def parse_args():
    parser = argparse.ArgumentParser(description="运行 AirSim 无人机穿洞策略")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="模型路径；默认优先使用 outputs/models 下的新模型",
    )
    parser.add_argument("--episodes", type=int, default=100, help="测试回合数")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="整个测试过程允许的最大步数",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备，例如 auto、cpu、cuda",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用随机动作采样；默认使用确定性动作",
    )
    return parser.parse_args()


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_model_path(model_path):
    if model_path is not None:
        candidates = [model_path, model_path.with_suffix(".zip")]
    else:
        candidates = [
            PROJECT_ROOT / "outputs" / "models" / "ppo_navigation_policy.zip",
            PROJECT_ROOT / "saved_policy" / "ppo_navigation_policy.zip",
        ]

    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("没有找到可用模型，请使用 --model 指定模型路径")


def infer_frame_stack(model):
    observation_shape = tuple(model.observation_space.shape)
    if len(observation_shape) != 3 or observation_shape[1:] != IMAGE_SHAPE[:2]:
        raise ValueError(
            "模型观测形状 %s 与当前 50×50 图像环境不兼容" % (observation_shape,)
        )
    channels = observation_shape[0]
    if channels % IMAGE_SHAPE[2] != 0:
        raise ValueError("模型输入通道数 %d 不是 RGB 通道数的整数倍" % channels)
    return channels // IMAGE_SHAPE[2]


def create_test_env(env_config, frame_stack):
    def make_env():
        env = gym.make(
            "scripts:test-env-v0",
            ip_address="127.0.0.1",
            image_shape=IMAGE_SHAPE,
            env_config=env_config,
            seed=42,
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="first")
    return env


def print_summary(distances):
    if not distances:
        print("没有完成任何测试回合。")
        return

    distances = np.asarray(distances, dtype=np.float32)
    holes = np.floor(distances / 4.0).astype(np.int32)
    print("\n========== 最终测试结果 ==========")
    print("完成回合数：%d" % len(distances))
    print("平均飞行距离：%.2f m" % float(np.mean(distances)))
    print("飞行距离标准差：%.2f m" % float(np.std(distances)))
    print("最大穿洞数：%d" % int(np.max(holes)))
    print("平均穿洞数：%.2f" % float(np.mean(holes)))
    print("==================================")


def main():
    args = parse_args()
    config = load_config()
    model_path = resolve_model_path(args.model)

    model = PPO.load(model_path, device=args.device)
    frame_stack = infer_frame_stack(model)
    env = create_test_env(config["TrainEnv"], frame_stack)
    model.set_env(env)

    print("加载模型：%s" % model_path)
    print("自动识别图像帧堆叠：%d" % frame_stack)

    completed_episodes = 0
    total_steps = 0
    obs = env.reset()

    try:
        while (
            completed_episodes < args.episodes
            and total_steps < args.max_steps
        ):
            action, _ = model.predict(
                obs,
                deterministic=not args.stochastic,
            )
            obs, _, dones, _ = env.step(action)
            total_steps += 1
            if bool(dones[0]):
                completed_episodes += 1

        distances = env.get_attr("agent_traveled")[0]
        print_summary(distances)

        if completed_episodes < args.episodes:
            print(
                "提示：达到最大步数 %d，仅完成 %d/%d 个回合。"
                % (args.max_steps, completed_episodes, args.episodes)
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
