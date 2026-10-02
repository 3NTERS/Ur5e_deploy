# UR5e 抬臂与夹爪联调模型部署手册

适用日期：2026-10-03

任务：`Ur5eRobotiqLiftGripper`

用途：无物体条件下完成“打开夹爪 → 关闭夹爪 → 抬臂 → 打开夹爪”的接口联调。

> 本模型只用于测试和联调，不是抓取策略。不得在人员进入机器人工作空间、急停不可达、工具负载未配置或安全参数未经现场风险评估时运行。现场安全配置和停止回路优先于本文档及策略输出。

## 1. 交付文件

部署内容统一位于 `/home/liang/Workspace/Ur5e_deploy`：

| 文件 | 用途 |
| --- | --- |
| `resources/models/ur5e_lift_gripper/ur5e_lift_gripper.onnx` | 真机和 MuJoCo 使用的确定性 `69 → 7` 策略 |
| `resources/models/ur5e_lift_gripper/ur5e_lift_gripper_distilled.pth` | Isaac Gym 回放及后续续训 |
| `resources/models/ur5e_lift_gripper/ur5e_lift_gripper.meta.yaml` | 观测、关节顺序、动作和时序定义 |
| `scripts/replay_ur5e_lift_gripper_isaacgym.sh` | Isaac Gym 单环境可视化回放入口 |
| `mujoco_sim/replay_ur5e_lift_gripper.py` | MuJoCo ONNX 回放和验收脚本 |
| `resources/assets/robots/ur5e_robotiq_2f85/ur5e_lift_gripper.xml` | MuJoCo UR5e + Robotiq 2F-85 模型 |

部署前必须检查文件摘要：

```bash
cd /home/liang/Workspace/Ur5e_deploy
sha256sum \
  resources/models/ur5e_lift_gripper/ur5e_lift_gripper_distilled.pth \
  resources/models/ur5e_lift_gripper/ur5e_lift_gripper.onnx
```

期望值：

```text
f1bb0be4cf6b8e7b138785166a765dc5e993a325fc92a4fab6d04b249944a23b  ur5e_lift_gripper_distilled.pth
8b8c6e7c34877a6d0fd3c30ae4731761def7dd5120f41406face07f042b28276  ur5e_lift_gripper.onnx
```

任一摘要不一致时停止部署，重新确认文件来源。

## 2. 已验证的软件环境

```text
Python          3.7
PyTorch         1.8.1+cu111
MuJoCo          2.3.6
ONNX Runtime    1.14.1
策略控制频率    60 Hz（dt = 0.01667 s）
```

进入环境：

```bash
source /home/liang/miniconda3/bin/activate rlgpu
cd /home/liang/Workspace/Ur5e_deploy
```

快速检查：

```bash
python - <<'PY'
import mujoco, onnxruntime
print("mujoco", mujoco.__version__)
print("onnxruntime", onnxruntime.__version__)
PY
```

## 3. 部署前仿真复验

### 3.1 Isaac Gym 可视化

```bash
./scripts/replay_ur5e_lift_gripper_isaacgym.sh
```

通过标准：输出 `LIFT_GRIPPER_EVAL`，并满足：

- `success_rate = 1.0`
- `worst_final_arm_error_rad <= 0.05`
- `worst_final_open_error_rad <= 0.05`
- `mean_max_tcp_lift_m >= 0.12`

本次验证结果：机械臂误差 `0.01408 rad`，夹爪误差 `1.72e-06 rad`，TCP 抬升 `0.16312 m`。

### 3.2 MuJoCo 可视化

```bash
python -m mujoco_sim.replay_ur5e_lift_gripper --realtime
```

无显示器环境可用：

```bash
python -m mujoco_sim.replay_ur5e_lift_gripper --headless
```

通过标准：进程返回 0，输出包含四个阶段及 `"success": true`。

本次验证结果：机械臂误差 `0.02361 rad`，夹爪最终误差 `0.00593 rad`，TCP 抬升 `0.14459 m`，最终连续成功 20 步。

## 4. 固定动作时序

一个 episode 为 240 个策略周期，约 4.0 秒。`progress` 必须严格从 0 增加到 239：

| progress | 时间 | 机械臂目标 | 夹爪目标 |
| ---: | ---: | --- | --- |
| 0–29 | 0.00–0.50 s | 初始位姿 | 打开 |
| 30–74 | 0.50–1.25 s | 初始位姿 | 关闭 |
| 75–194 | 1.25–3.25 s | 抬升位姿 | 保持关闭 |
| 195–239 | 3.25–4.00 s | 抬升位姿 | 打开 |

初始关节角：

```text
[-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0] rad
```

最终关节角：

```text
[-1.5708, -1.2000, 1.2000, -1.5708, -1.5708, 0.0] rad
```

只有肩部抬升关节和肘关节发生主要运动。启动前实测关节角与初始角的最大偏差必须不超过 `0.05 rad`。

## 5. ONNX 接口

### 5.1 输入

输入名称为 `observation`，形状为 `[batch, 69]`，类型 `float32`。

| 切片 | 内容 |
| --- | --- |
| `[0:12]` | 6 个 UR5e + 6 个 Robotiq 关节位置，按上下限缩放到 `[-1,1]` |
| `[12:24]` | 同顺序关节速度，单位 rad/s |
| `[24:27]` | 夹爪掌心中心世界坐标，单位 m |
| `[27:37]` | 掌心四元数 `xyzw`、线速度、角速度 |
| `[37:47]` | 固定为 0 |
| `[47:53]` | 左右指尖相对掌心位置 |
| `[53:66]` | 固定为 0 |
| `[66]` | `log(progress / 10 + 1)` |
| `[67:69]` | 固定为 0 |

当前交付模型经过固定时序蒸馏，实际只依赖 `observation[66]`；其他输入仍应保留，以保持 69 维接口并兼容后续模型。

### 5.2 输出

输出名称为 `action`，形状为 `[batch, 7]`，范围 `[-1,1]`：

- `action[0:6]`：UR5e 六关节增量命令。
- `action[6]`：Robotiq 同步开合命令，`-1` 为打开，`+1` 为关闭。

机械臂目标更新：

```python
q_target = clip(q_target_previous + 0.01667 * action[0:6], joint_limits)
```

不可用当前实测关节角替代 `q_target_previous` 做每步积分，否则跟踪误差会改变整条轨迹。实测角只用于监控和急停判断。

夹爪仿真主关节目标：

```python
finger_joint_target = 0.36 * (action[6] + 1.0)  # 0.00 ... 0.72 rad
```

真机 Robotiq 位置字节建议先保守映射到 `0...230`，与仿真 90% 行程一致：

```python
rPR = round(115.0 * (clip(action[6], -1, 1) + 1.0))
```

确认机械限位、指尖间隙和夹持力后，才可由现场负责人决定是否改成全行程 `0...255`。

## 6. 真机控制循环参考

仓库目前提供 Isaac Gym 和 MuJoCo 适配器，没有绑定现场 IP、RTDE/ROS 驱动及 Robotiq 通信方式的真机执行器。必须由现场已有驱动完成最后一层连接，不要把 ONNX 输出直接当作力矩或速度发送。

控制结构：

```python
assert robot_is_at_initial_pose(tolerance=0.05)
assert gripper_is_open()
q_target = INITIAL_Q.copy()

for progress in range(240):
    tick = monotonic_time()
    obs = build_69_float32_observation(progress)
    action = clip(onnx_session.run(None, {"observation": obs})[0][0], -1, 1)

    q_target += 0.01667 * action[:6]
    q_target = clip_to_deployment_corridor(q_target)
    gripper_request = round(115.0 * (action[6] + 1.0))

    send_joint_position_target(q_target)
    send_gripper_position(gripper_request)
    enforce_watchdogs()
    wait_until(tick + 0.01667)
```

策略以 60 Hz 更新目标。若现场 UR 驱动要求更高的伺服更新频率，应由驱动层在两个策略目标之间保持或插值，并遵守该驱动规定的周期；不要简单降低驱动自身的必需更新频率。

每次推理前检查：

- ONNX 输入形状严格为 `(1,69)` 且为 `float32`。
- 数值全部有限，不含 NaN/Inf。
- `progress` 不跳步、不复用旧 episode 的值。
- 输出裁剪到 `[-1,1]`。
- 通信、机器人安全状态和夹爪状态正常。

## 7. 明天的分阶段上线流程

### A. 上电前

1. 清空工作空间，移除物体和不需要的工装。
2. 核对 TCP、负载、重心、夹爪安装方向和线缆余量。
3. 确认示教器急停、外部急停和防护停止可用，操作人始终能触达急停。
4. 确认安全平面、关节限制、TCP 速度/力/动量限制由现场风险评估确定。
5. 记录当前软件版本、模型 SHA256、机器人序列号和安全配置校验值。

### B. 无使能干跑

1. 连接机器人和夹爪，但不使能运动。
2. 连续运行 240 次 ONNX 推理并记录时间戳、progress、action 和积分目标。
3. 检查目标始终位于下述部署走廊，无 NaN、突跳或丢周期。

建议的软件部署走廊是初始与目标路径外加 `0.05 rad`：

```text
J1 [-1.6208, -1.5208]
J2 [-1.6208, -1.1500]
J3 [ 1.1500,  1.6208]
J4 [-1.6208, -1.5208]
J5 [-1.6208, -1.5208]
J6 [-0.0500,  0.0500]
```

### C. 低速单步

1. 使用手动/示教模式到达初始位姿，夹爪打开。
2. 首次测试使用现场允许的最低速度比例，并采用保持使能的单步方式。
3. 分别只执行到 progress 29、74、194、239，每段结束后人工检查姿态和线缆。
4. 任何方向与仿真不一致，立即停止；优先检查关节顺序、弧度/角度和夹爪方向。

### D. 连续运行

1. 确认低速单步四阶段全部正确后，才允许完整运行 0–239。
2. 第一次完整运行仍保持低速和空工作空间。
3. 至少成功重复 3 次后，再由现场负责人决定是否提高速度比例。
4. 本任务不需要物体；不要在首次部署时加入抓取物。

## 8. 必须触发停止的条件

下列任一条件出现时，停止发送新目标并使用现场已经验证的停止方式：

- 实测关节与命令目标最大误差超过 `0.15 rad`，持续 3 个策略周期。
- 任一目标超出部署走廊。
- 单周期超过 `100 ms`、时钟回退或 progress 不连续。
- ONNX 推理报错，输出出现 NaN/Inf 或形状不是 7。
- 机器人进入保护停止、急停、故障或非预期安全模式。
- TCP 或夹爪向与仿真相反的方向运动。
- 夹爪碰撞、线缆拉扯、异常声音、电流或温升。
- 操作人员无法持续观察机器人或急停不可达。

停止后不得自动清除急停或保护停止，不得从中间 progress 自动续跑。人工确认原因后，将机器人安全返回初始位姿，重新从 progress 0 开始。

## 9. 日志要求

每个 60 Hz 周期至少记录：

```text
monotonic_timestamp
progress
measured_joint_position[6]
measured_joint_velocity[6]
commanded_joint_target[6]
gripper_request
policy_action[7]
onnx_inference_time_ms
robot_safety_state
driver_status
```

部署验收记录应包含：

- 4 个阶段切换时间是否为 30、75、195。
- 最终最大关节误差是否不超过 `0.05 rad`。
- 夹爪关闭及最终打开是否均达到现场定义的容差。
- TCP 抬升方向是否正确，实际高度是否至少 `0.12 m`。
- 240 步内是否发生通信丢包、保护停止或 watchdog 触发。

## 10. 常见故障

| 现象 | 优先检查 |
| --- | --- |
| 机械臂反向运动 | 关节名称顺序、驱动关节符号、弧度/角度单位 |
| 机械臂几乎不动 | 是否把增量 action 误当成绝对关节角；驱动是否处于位置控制 |
| 轨迹逐步漂移 | 是否用实测角替代上一目标积分；控制周期是否不是 0.01667 s |
| 夹爪开合相反 | Robotiq `rPR` 方向和驱动配置 |
| ONNX 输出不随机器人状态变化 | 当前蒸馏模型只依赖 progress，这是预期行为 |
| MuJoCo 最终误差大于 Isaac Gym | 两套位置执行器动力学不同；以各自 `0.05 rad` 验收门槛判断 |
| Isaac Gym 无窗口 | 检查 `DISPLAY`、显卡驱动，并确认 `headless=False` |
| MuJoCo 无窗口 | 使用 `--headless` 先验收；检查 GLFW/X11 环境 |

## 11. 安全资料

UR5e 的安全参数、Reduced 配置、安全 I/O 和停止类别以现场 PolyScope 软件版本对应的官方手册为准：

- [Universal Robots UR5e 用户手册（SW 5.23）](https://www.universal-robots.com/manuals/EN/PDF/SW5_23/user-manual-UR5e-PDF_online/710-965-00_UR5e_User_Manual_en_Global.pdf)

Reduced 是一组较低的安全限制配置，不等同于完成了应用风险评估。急停和防护停止的复位、重新启动必须遵守现场安全回路和官方手册；本脚本不执行安全复位。

Robotiq 2F-85 的安装、夹持力、行程和维护以现场对应硬件版本的官方说明书为准。首次联调使用空夹爪和保守行程，不夹持物体。
