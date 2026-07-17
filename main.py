import argparse
import time
from collections import deque
from pathlib import Path

import gym
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "scripts" / "config.yml"
IMAGE_SHAPE = (50, 50, 3)


class TrainingSuccessCallback(BaseCallback):
    """根据训练回合的滑动成功率保存候选最优模型，不额外重置环境。"""

    def __init__(
        self,
        save_path,
        window_size=100,
        min_episodes=50,
        verbose=1,
    ):
        super().__init__(verbose=verbose)
        self.save_path = str(save_path)
        self.successes = deque(maxlen=int(window_size))
        self.min_episodes = int(min_episodes)
        self.best_success_rate = -1.0

    def _on_step(self):
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        episode_finished = False

        for done, info in zip(dones, infos):
            if bool(done):
                self.successes.append(bool(info.get("is_success", False)))
                episode_finished = True

        if episode_finished and len(self.successes) >= self.min_episodes:
            success_rate = sum(self.successes) / len(self.successes)
            self.logger.record("rollout/window_success_rate", success_rate)
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                self.model.save(self.save_path)
                if self.verbose:
                    print(
                        "训练滑动成功率刷新为 %.1f%%，已保存候选最优模型"
                        % (100.0 * success_rate)
                    )
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="使用 PPO 训练 AirSim 无人机穿洞策略")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="本次训练步数；默认读取 scripts/config.yml",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--device",
        default="auto",
        help="训练设备，例如 auto、cpu、cuda",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=None,
        help="堆叠图像帧数；默认读取配置，续训时默认沿用模型输入",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从已有 PPO 模型或检查点继续训练",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="模型、日志和检查点保存目录",
    )
    return parser.parse_args()


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def linear_schedule(initial_value):
    """随训练进度将学习率从初始值线性衰减到 0。"""

    initial_value = float(initial_value)

    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


def resolve_model_path(path):
    path = path.expanduser().resolve()
    if path.exists():
        return path
    zip_path = path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    raise FileNotFoundError("找不到续训模型：%s" % path)


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


def create_training_env(env_config, seed, frame_stack, monitor_path):
    def make_env():
        env = gym.make(
            "scripts:airsim-env-v0",
            ip_address="127.0.0.1",
            image_shape=IMAGE_SHAPE,
            env_config=env_config,
            seed=seed,
        )
        return Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=(
                "is_success",
                "termination_reason",
                "target_distance",
                "forward_distance",
                "invalid_image_count",
            ),
        )

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack, channels_order="first")
    return env


def build_new_model(env, training_config, seed, device, tensorboard_dir):
    return PPO(
        "CnnPolicy",
        env,
        learning_rate=linear_schedule(training_config["learning_rate"]),
        n_steps=int(training_config["n_steps"]),
        batch_size=int(training_config["batch_size"]),
        n_epochs=int(training_config["n_epochs"]),
        gamma=float(training_config["gamma"]),
        gae_lambda=float(training_config["gae_lambda"]),
        clip_range=float(training_config["clip_range"]),
        ent_coef=float(training_config["ent_coef"]),
        vf_coef=float(training_config["vf_coef"]),
        max_grad_norm=float(training_config["max_grad_norm"]),
        target_kl=float(training_config["target_kl"]),
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(tensorboard_dir),
    )


def main():
    args = parse_args()
    config = load_config()
    training_config = config["Training"]

    output_dir = args.output_dir.expanduser().resolve()
    model_dir = output_dir / "models"
    checkpoint_dir = output_dir / "checkpoints"
    monitor_dir = output_dir / "monitor"
    tensorboard_dir = output_dir / "tensorboard"
    for directory in (
        model_dir,
        checkpoint_dir,
        monitor_dir,
        tensorboard_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = "ppo_navigation_%s" % time.strftime("%Y%m%d_%H%M%S")
    resume_model = None

    if args.resume is not None:
        resume_path = resolve_model_path(args.resume)
        resume_model = PPO.load(resume_path, device=args.device)
        model_frame_stack = infer_frame_stack(resume_model)
        if args.frame_stack is not None and args.frame_stack != model_frame_stack:
            raise ValueError(
                "续训模型使用 %d 帧输入，不能改为 %d 帧"
                % (model_frame_stack, args.frame_stack)
            )
        frame_stack = model_frame_stack
    else:
        frame_stack = (
            args.frame_stack
            if args.frame_stack is not None
            else int(training_config["frame_stack"])
        )

    if frame_stack < 1:
        raise ValueError("frame_stack 必须大于等于 1")

    env = create_training_env(
        config["TrainEnv"],
        seed=args.seed,
        frame_stack=frame_stack,
        monitor_path=monitor_dir / run_name,
    )

    if resume_model is not None:
        model = resume_model
        model.set_env(env)
        model.tensorboard_log = str(tensorboard_dir)
        print("从检查点继续训练，累计步数：%d" % model.num_timesteps)
    else:
        model = build_new_model(
            env,
            training_config,
            seed=args.seed,
            device=args.device,
            tensorboard_dir=tensorboard_dir,
        )

    total_timesteps = (
        args.timesteps
        if args.timesteps is not None
        else int(training_config["total_timesteps"])
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=int(training_config["checkpoint_freq"]),
        save_path=str(checkpoint_dir),
        name_prefix=run_name,
    )
    success_callback = TrainingSuccessCallback(
        save_path=model_dir / ("%s_best_training_policy" % run_name),
        window_size=int(training_config["success_window"]),
        min_episodes=int(training_config["min_success_episodes"]),
    )
    callbacks = CallbackList([checkpoint_callback, success_callback])

    print("图像帧堆叠：%d" % frame_stack)
    print("本次训练步数：%d" % total_timesteps)
    print("输出目录：%s" % output_dir)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=resume_model is None,
        )
        final_model_path = model_dir / "ppo_navigation_policy"
        model.save(final_model_path)
        print("训练完成，模型已保存到：%s.zip" % final_model_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
