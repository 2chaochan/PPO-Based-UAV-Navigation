# 基于 PPO 的四旋翼无人机自主穿洞

本项目使用 **PPO（Proximal Policy Optimization）** 训练四旋翼无人机，仅依靠前置 RGB 相机在 AirSim/Unreal Engine 走廊场景中连续穿越圆形洞口。

策略网络在推理时不读取无人机位置、洞口坐标或深度图。仿真器中的真实位置只用于训练阶段计算奖励，因此这是一个“训练时使用特权信息、推理时纯视觉控制”的强化学习任务。

> 项目依赖 Windows 版本的 Unreal/AirSim 场景。仓库只包含训练代码和预训练权重，不包含 `TrainEnv`、`TestEnv` 场景可执行文件。

## 效果展示
![演示](./demo/demo.gif)

## 1. 项目原理

### 1.1 观测空间

- 前置相机 RGB 图像
- 单帧尺寸：`50 × 50 × 3`
- 新训练默认堆叠 4 帧，模型输入为 `12 × 50 × 50`
- 图像进入 Stable Baselines3 的 `CnnPolicy/NatureCNN`

堆叠多帧后，策略可以感知洞口在画面中的相对运动和接近速度。仓库内原有预训练模型仍是单帧输入，推理脚本会自动识别。

### 1.2 动作空间

动作空间包含 9 个离散动作。无人机始终以 `0.4 m/s` 向前飞行，策略只控制横向和竖向速度：

| 竖直方向 \ 横向方向 | 左 | 中 | 右 |
|---|---:|---:|---:|
| 上 | 0 | 1 | 2 |
| 保持 | 3 | 4 | 5 |
| 下 | 6 | 7 | 8 |

AirSim 使用 NED 坐标系，因此 `z < 0` 表示向上。

### 1.3 训练场景

训练场景包含 9 个区域，相邻墙面间隔 4 米。每个训练回合会：

1. 随机选择一个墙面区域；
2. 在墙前的 `y-z` 平面随机初始化无人机；
3. 使用 RGB 图像调整无人机与洞口的相对位置；
4. 穿过当前墙面、发生碰撞、丢失洞口或超时后结束回合。

训练阶段学习的是“穿过单面墙”的局部视觉策略。测试阶段则重复使用同一个策略，连续穿越多面墙。

## 2. 奖励设计

奖励参数位于 [`scripts/config.yml`](scripts/config.yml)。

当前奖励由以下部分构成：

- 靠近洞心的距离进度奖励；
- 随无人机接近墙面逐渐增强的连续对准奖励；
- 少量向前运动奖励；
- 每步时间惩罚；
- 成功穿洞：`+100`；
- 撞击墙面：`-100`；
- 洞口离开相机视野：`-50`；
- 回合超时：`-30`。

相较原始实现，新奖励有以下变化：

- 修复了回合初始化时 AirSim Z 轴符号不一致的问题；
- 用平滑连续函数替代 `0.30/0.45m` 两个硬阈值；
- 提高成功奖励，使“真正穿洞”成为主要优化目标；
- 增加回合步数上限，避免异常状态导致训练卡住。

## 3. 目录结构

```text
.
├── main.py                         # PPO 训练入口
├── policy_run.py                   # 模型测试入口
├── requirements.txt                # Python 依赖
├── saved_policy/
│   └── ppo_navigation_policy.zip   # 原始单帧预训练模型
└── scripts/
    ├── __init__.py                 # 注册 Gym 环境
    ├── airsim_env.py               # 动作、观测、奖励和测试统计
    ├── config.yml                  # 环境、奖励和 PPO 参数
    └── airsim/                     # 项目内置 AirSim Python 客户端
```

训练后会生成：

```text
outputs/
├── checkpoints/   # 定期检查点
├── models/        # 最终模型和按训练成功率保存的候选最优模型
├── monitor/       # 每回合奖励、成功率和终止原因
└── tensorboard/   # TensorBoard 日志
```

## 4. 环境安装

项目原始依赖栈较旧，建议使用独立的 Python 3.8 环境：

```powershell
conda create -n ppo_drone python=3.8
conda activate ppo_drone
pip install -r requirements.txt
```

不建议直接使用 Python 3.11、Gym 0.26 或 Stable Baselines3 2.x 运行本项目，因为 Gym 的 `reset/step` 接口和模型保存格式已经发生变化。

如果使用 NVIDIA GPU，请安装与本机 CUDA 驱动匹配的 PyTorch。训练脚本默认使用 `--device auto`，会自动选择可用设备。

## 5. AirSim 配置

编辑：

```text
文档\AirSim\settings.json
```

推荐配置：

```json
{
  "SettingsVersion": 1.2,
  "LocalHostIp": "127.0.0.1",
  "SimMode": "Multirotor",
  "ClockSpeed": 20,
  "ViewMode": "SpringArmChase",
  "Vehicles": {
    "drone0": {
      "VehicleType": "SimpleFlight",
      "X": 0.0,
      "Y": 0.0,
      "Z": 0.0,
      "Yaw": 0.0
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      {
        "ImageType": 0,
        "Width": 50,
        "Height": 50,
        "FOV_Degrees": 120
      }
    ]
  }
}
```

训练时可以使用较高的 `ClockSpeed`。如果出现物理抖动、控制指令跳变或大量异常碰撞，建议先降到 `5～10` 验证稳定性。测试和录像时建议设为 `1`。

## 6. 下载仿真场景

从原项目 [Releases](https://github.com/bilalkabas/PPO-based-Autonomous-Navigation-for-Quadcopters/releases) 下载：

- `TrainEnv.zip`：训练场景；
- `TestEnv.zip`：测试场景。

解压后先启动对应的 Unreal 可执行文件，等待场景完全加载，再运行 Python 脚本。

## 7. 开始训练

启动 `TrainEnv` 后，在项目根目录执行：

```powershell
python main.py
```

默认训练参数：

| 参数 | 默认值 |
|---|---:|
| 训练步数 | 500,000 |
| 帧堆叠 | 4 |
| Rollout 步数 | 1,024 |
| Batch size | 64 |
| Epochs | 10 |
| 初始学习率 | 0.0003，线性衰减 |
| 折扣因子 `gamma` | 0.99 |
| GAE `lambda` | 0.95 |
| PPO clip | 0.2 |
| 熵系数 | 0.01 |
| Target KL | 0.03 |
| 检查点间隔 | 10,000 步 |
| 成功率滑动窗口 | 100 回合 |

也可以覆盖部分参数：

```powershell
python main.py --timesteps 1000000 --device cuda
```

### 7.1 从检查点继续训练

```powershell
python main.py `
  --resume outputs/checkpoints/ppo_navigation_时间戳_100000_steps.zip `
  --timesteps 300000
```

续训时会自动读取模型需要的帧堆叠数量。不能把已经训练好的单帧模型直接改成四帧输入；如果希望使用四帧，应从头训练新模型。

### 7.2 查看训练曲线

```powershell
tensorboard --logdir outputs/tensorboard
```

建议重点观察：

- `rollout/success_rate`：训练成功率；
- `rollout/ep_rew_mean`：平均回合奖励；
- `rollout/ep_len_mean`：平均回合长度；
- `train/approx_kl`：PPO 每次更新幅度；
- `train/entropy_loss`：策略探索程度；
- `train/explained_variance`：价值网络拟合情况。

不要只看平均奖励。奖励提高但成功率不提高，通常说明策略在利用稠密奖励，却没有稳定穿洞。

训练脚本还会依据最近 100 个训练回合的滑动成功率保存
`outputs/models/时间戳_best_training_policy.zip`。它不会重置仿真环境，但训练动作带有探索噪声，所以仍应在 `TestEnv` 中比较候选模型和最终模型。

## 8. 运行模型

启动 `TestEnv` 后执行：

```powershell
python policy_run.py --episodes 100
```

推理脚本会优先加载：

```text
outputs/models/ppo_navigation_policy.zip
```

如果新模型不存在，则加载：

```text
saved_policy/ppo_navigation_policy.zip
```

指定其他模型：

```powershell
python policy_run.py --model outputs/checkpoints/某个检查点.zip --episodes 100
```

脚本会输出平均飞行距离、标准差、最大穿洞数和逐回合计算的平均穿洞数。

## 9. 训练效果不佳时的排查顺序

### 9.1 先确认图像输入

相机必须稳定返回 `50×50×3` 的 RGB 图像。如果 AirSim 配置分辨率不一致，环境会返回全零图像，此时策略无法学习。

Monitor 日志中的 `invalid_image_count` 会记录该回合收到的异常图像数量；遇到奖励长期不变化时，应优先保存几帧实际观测进行确认。

### 9.2 使用由易到难的初始化范围

如果从头训练一直没有成功回合，可以先修改：

```yaml
random_start_range: 0.5
```

训练出基础对准能力后，再恢复为：

```yaml
random_start_range: 1.0
```

然后从检查点续训。这是一种简单的课程学习方式。

### 9.3 检查控制与物理速度

如果无人机每步位移明显大于预期、在墙前剧烈震荡或碰撞时间戳异常：

1. 将 `ClockSpeed` 降到 `1`；
2. 检查单个动作是否持续约 1 秒仿真时间；
3. 再逐步提高 `ClockSpeed`。

### 9.4 不要用同一个 AirSim 环境同时训练和评估

原始代码的 `EvalCallback` 会重置正在训练的同一架无人机，导致 PPO 保存的观测与实际仿真状态错位。当前版本已移除这一逻辑，改为定期保存检查点。

如果需要训练中独立评估，应启动独立的 AirSim 实例、端口或车辆，不能与训练环境共享同一状态。

### 9.5 先关注成功率，再调奖励

推荐调整顺序：

1. 确认图像和动作方向正确；
2. 确认至少偶尔出现成功回合；
3. 观察成功率是否持续上升；
4. 最后再微调 `alignment_scale`、`progress_scale` 和惩罚值。

一次只修改一个主要参数，并至少训练数万步后再比较结果。

## 10. 预训练模型说明

仓库内原始模型使用：

- 单帧 RGB 输入；
- 9 个离散动作；
- Stable Baselines3 1.1 保存格式；
- 模型归档记录的当前训练计数为 51,200 步。

README 原版本描述为训练 280,000 步，与模型文件中的计数不一致，可能是多阶段训练时重置了计数，也可能是提交了不同阶段的权重。进行实验对比时，应以实际模型文件和 Monitor/TensorBoard 日志为准。

## 11. 许可证

本项目使用 [GNU AGPL 3.0](LICENSE) 许可证。
