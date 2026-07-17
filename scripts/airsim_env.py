import math

import gym
import numpy as np

from . import airsim


class AirSimDroneEnv(gym.Env):
    """AirSim 单目视觉穿洞训练环境（兼容 Gym 0.21 接口）。"""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, ip_address, image_shape, env_config, seed=None):
        self.image_shape = tuple(image_shape)
        self.sections = env_config["sections"]

        self.speed = float(env_config.get("speed", 0.4))
        self.action_duration = float(env_config.get("action_duration", 1.0))
        self.settle_duration = float(env_config.get("settle_duration", 0.05))
        self.random_start_range = float(env_config.get("random_start_range", 1.0))
        self.success_distance = float(env_config.get("success_distance", 3.7))
        self.max_episode_steps = int(env_config.get("max_episode_steps", 20))
        self.camera_fov_degrees = float(env_config.get("camera_fov_degrees", 120.0))
        self.hole_radius = float(env_config.get("hole_radius", 0.30))
        self.reward_config = env_config.get("reward", {})

        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.image_shape,
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Discrete(9)

        self.random_start = True
        self.collision_time = 0
        self.step_count = 0
        self.target_pos_idx = 0
        self.agent_start_pos = 0.0
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.target_dist_prev = 0.0
        self.agent_x_prev = 0.0
        self.last_image_valid = True
        self.invalid_image_count = 0
        self._last_obs = np.zeros(self.image_shape, dtype=np.uint8)
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.do_action(action)
        self.step_count += 1

        collision = self.is_collision()
        obs = self.get_rgb_image()
        reward, done, reward_info = self.compute_reward(collision)
        info = {
            "collision": collision,
            "image_valid": self.last_image_valid,
            "invalid_image_count": self.invalid_image_count,
            **reward_info,
        }
        return obs, float(reward), bool(done), info

    def reset(self):
        self.setup_flight()
        return self.get_rgb_image()

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise ValueError("AirSimDroneEnv 仅支持 rgb_array 渲染模式")
        return self._last_obs

    def close(self):
        try:
            self.drone.armDisarm(False)
            self.drone.enableApiControl(False)
        except Exception:
            # 仿真器已经退出时，关闭环境不应再次导致训练脚本报错。
            pass

    def _randint(self, high):
        if hasattr(self.np_random, "integers"):
            return int(self.np_random.integers(high))
        return int(self.np_random.randint(high))

    def _sample_start_position(self):
        start = self.np_random.uniform(
            low=-self.random_start_range,
            high=self.random_start_range,
            size=2,
        )
        return float(start[0]), float(start[1])

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        if self.random_start:
            self.target_pos_idx = self._randint(len(self.sections))
        else:
            self.target_pos_idx = 0

        section = self.sections[self.target_pos_idx]
        self.agent_start_pos = float(section["offset"][0])
        self.target_pos = np.asarray(section["target"], dtype=np.float32)

        y_pos, z_pos = self._sample_start_position()
        pose = airsim.Pose(
            airsim.Vector3r(self.agent_start_pos, y_pos, z_pos)
        )
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

        # 用一个很短的零速度指令稳定飞机，避免 reset 后遗留的异步指令继续生效。
        if self.settle_duration > 0:
            self.drone.moveByVelocityAsync(
                vx=0,
                vy=0,
                vz=0,
                duration=self.settle_duration,
            ).join()

        # AirSim 采用 NED 坐标系，配置中的第二个洞口坐标按“向上为正”保存。
        self.target_dist_prev = float(
            np.linalg.norm(np.array([y_pos, -z_pos]) - self.target_pos)
        )
        self.agent_x_prev = self.agent_start_pos
        self.step_count = 0
        self.invalid_image_count = 0
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

    def do_action(self, selected_action):
        action = int(np.asarray(selected_action).item())
        if not self.action_space.contains(action):
            raise ValueError("动作必须是 0 到 8 之间的整数，当前值为 %s" % action)

        row, column = divmod(action, 3)
        vy = (column - 1) * self.speed
        vz = (row - 1) * self.speed

        # x 方向保持固定前进速度，策略只负责横向和竖向对准洞口。
        self.drone.moveByVelocityBodyFrameAsync(
            self.speed,
            vy,
            vz,
            duration=self.action_duration,
        ).join()

        if self.settle_duration > 0:
            self.drone.moveByVelocityAsync(
                vx=0,
                vy=0,
                vz=0,
                duration=self.settle_duration,
            ).join()

    def compute_reward(self, collision):
        x, y, z = self.drone.simGetVehiclePose().position
        target_dist_curr = float(
            np.linalg.norm(np.array([y, -z]) - self.target_pos)
        )
        forward_distance = max(0.0, float(x) - self.agent_start_pos)
        forward_delta = max(0.0, float(x) - self.agent_x_prev)
        self.agent_x_prev = float(x)

        progress_scale = float(self.reward_config.get("progress_scale", 20.0))
        alignment_scale = float(self.reward_config.get("alignment_scale", 2.0))
        alignment_sigma = float(self.reward_config.get("alignment_sigma", 0.45))
        forward_scale = float(self.reward_config.get("forward_scale", 0.5))
        step_penalty = float(self.reward_config.get("step_penalty", 0.05))

        distance_progress = self.target_dist_prev - target_dist_curr
        progress_ratio = np.clip(
            forward_distance / self.success_distance,
            0.0,
            1.0,
        )
        alignment = math.exp(
            -0.5 * (target_dist_curr / max(alignment_sigma, 1e-6)) ** 2
        )

        # 连续对准奖励在临近墙面时逐渐增强，避免原先阈值奖励造成突变。
        reward = (
            progress_scale * distance_progress
            + alignment_scale * alignment * (0.5 + 1.5 * progress_ratio)
            + forward_scale * forward_delta
            - step_penalty
        )
        self.target_dist_prev = target_dist_curr

        done = False
        is_success = False
        termination_reason = "running"

        if collision:
            reward = float(self.reward_config.get("collision_penalty", -100.0))
            done = True
            termination_reason = "collision"
        elif forward_distance >= self.success_distance:
            reward += float(self.reward_config.get("success_reward", 100.0))
            done = True
            is_success = True
            termination_reason = "success"
        elif self._hole_is_out_of_view(target_dist_curr, forward_distance):
            reward = float(self.reward_config.get("lost_target_penalty", -50.0))
            done = True
            termination_reason = "lost_target"
        elif self.step_count >= self.max_episode_steps:
            reward += float(self.reward_config.get("timeout_penalty", -30.0))
            done = True
            termination_reason = "timeout"

        info = {
            "is_success": is_success,
            "termination_reason": termination_reason,
            "target_distance": target_dist_curr,
            "forward_distance": forward_distance,
        }
        return reward, done, info

    def _hole_is_out_of_view(self, target_distance, forward_distance):
        remaining_distance = max(0.0, self.success_distance - forward_distance)
        half_fov_radians = math.radians(self.camera_fov_degrees / 2.0)
        visible_radius = remaining_distance * math.tan(half_fov_radians)
        return (target_distance - self.hole_radius) > visible_radius

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return current_collision_time != self.collision_time

    def get_rgb_image(self):
        request = airsim.ImageRequest(
            0,
            airsim.ImageType.Scene,
            False,
            False,
        )
        responses = self.drone.simGetImages([request])

        self.last_image_valid = False
        if responses:
            response = responses[0]
            expected_size = int(response.height) * int(response.width) * 3
            image_data = response.image_data_uint8
            if (
                expected_size > 0
                and image_data is not None
                and len(image_data) == expected_size
            ):
                try:
                    image = np.frombuffer(image_data, dtype=np.uint8).reshape(
                        response.height,
                        response.width,
                        3,
                    )
                    if image.shape == self.image_shape:
                        self._last_obs = image
                        self.last_image_valid = True
                except (TypeError, ValueError):
                    pass

        if not self.last_image_valid:
            self.invalid_image_count += 1
            self._last_obs = np.zeros(self.image_shape, dtype=np.uint8)
        return self._last_obs

    def get_depth_image(self, thresh=2.0):
        request = airsim.ImageRequest(
            1,
            airsim.ImageType.DepthPerspective,
            True,
            False,
        )
        responses = self.drone.simGetImages([request])
        depth_image = np.asarray(
            responses[0].image_data_float,
            dtype=np.float32,
        ).reshape(responses[0].height, responses[0].width)
        return np.minimum(depth_image, thresh)


class TestEnv(AirSimDroneEnv):
    """连续穿越测试场景，直到碰撞或达到安全步数上限。"""

    def __init__(self, ip_address, image_shape, env_config, seed=None):
        self.agent_traveled = []
        self.test_max_episode_steps = int(
            env_config.get("test_max_episode_steps", 500)
        )
        super().__init__(ip_address, image_shape, env_config, seed=seed)
        self.random_start = False

    def _sample_start_position(self):
        return 0.0, 0.0

    def compute_reward(self, collision):
        x, _, _ = self.drone.simGetVehiclePose().position
        flight_distance = max(0.0, float(x) - self.agent_start_pos)

        timed_out = self.step_count >= self.test_max_episode_steps
        done = bool(collision or timed_out)
        termination_reason = "running"
        if collision:
            termination_reason = "collision"
        elif timed_out:
            termination_reason = "timeout"

        if done:
            self.agent_traveled.append(flight_distance)
            completed_episodes = len(self.agent_traveled)
            if completed_episodes % 5 == 0:
                distances = np.asarray(self.agent_traveled, dtype=np.float32)
                holes = np.floor(distances / 4.0).astype(np.int32)
                print("---------------------------------")
                print("> 已完成回合:", completed_episodes)
                print("> 平均飞行距离: %.2f m" % float(np.mean(distances)))
                print("> 最大穿洞数:", int(np.max(holes)))
                print("> 平均穿洞数: %.2f" % float(np.mean(holes)))
                print("---------------------------------\n")

        info = {
            "is_success": False,
            "termination_reason": termination_reason,
            "flight_distance": flight_distance,
        }
        return 0.0, done, info
