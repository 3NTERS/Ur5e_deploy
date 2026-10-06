# UR5e + Robotiq 策略仿真与真机部署

本仓库用于将 Isaac Gym 中训练的 UR5e + Robotiq 2F-85 策略导出为 ONNX，先在
Isaac Gym 和 MuJoCo 中预览、完成 sim2sim 对齐，再通过 RTDE、Robotiq URCap socket、
RealSense 和 YOLO 运行真机策略。

当前真机主线任务是 `Ur5eRobotiqHoverGripper`：策略输入为 69 维 observation，输出为
7 维 action（6 个机械臂关节增量和 1 个夹爪绝对位置），策略周期为 `0.01667 s`，每回合
300 步。以下命令均在仓库根目录执行。

## 项目结构

- `mujoco_sim/`：MuJoCo adapter、轨迹运行器和 Isaac Gym/MuJoCo 对齐逻辑。
- `onnx_deploy/`：通用 ONNX Runtime 策略执行器，支持普通 MLP 以及带
  `hidden_state`/`cell_state` 的循环网络。
- `ur5e_comm/`：UR RTDE、Robotiq socket、RealSense、YOLO、状态投影和安全检查。
- `resources/assets/`：已手工确认的 UR5e、Robotiq 和 MuJoCo scene 资源。
- `resources/config/`：仿真和真机部署配置。
- `resources/models/`：ONNX、checkpoint、YOLO 权重和策略接口 metadata。
- `resources/calibration/`：眼在手上标定结果与采集记录。
- `resources/trajectories/`：Isaac Gym、MuJoCo 和真机轨迹。
- `scripts/`：训练、导出、预览、sim2sim、标定和真机运行入口。
- `tests/`：策略接口、观测、动作映射和仿真闭环测试。
- `docs/`：补充说明和历史方案。
- `offline_artifacts/`：未纳入 Git 的模型、标定、轨迹等文件的分组压缩包、恢复脚本与说明；
  生成的 `archives/` 不会上传。

当前 hover-gripper 部署主要使用：

| 阶段 | 入口 |
| --- | --- |
| Isaac Gym 预览 | `scripts/replay_ur5e_hover_isaacgym.sh` |
| 导出 Isaac Gym 参考轨迹 | `scripts/export_ur5e_isaac_reference.sh` |
| MuJoCo sim2sim | `scripts/run_ur5e_hover_sim2sim.sh` |
| 腕部相机标定 | `scripts/calibrate_eye_on_hand.py` |
| YOLO/深度检查 | `scripts/inspect_yolo_realsense.py` |
| 真机环境只读预览 | `scripts/view_ur5e_hover_camera_object_mujoco.py` |
| 真机策略 | `scripts/run_ur5e_hover_gripper_policy.py` |
| 真机轨迹回放 | `scripts/replay_ur5e_hover_real_mujoco.py` |

`scripts/run_ur5e_policy.py`、`resources/config/ur5e_deploy.yaml` 是原有 grasp/双相机链路；
Rokae 相关入口也继续保留，但都不是本 README 的 hover-gripper 真机主线。

## 环境

项目复用 Isaac Gym 使用的 `rlgpu` Conda 环境。基础仿真依赖为 MuJoCo 2.3.6、
ONNX 1.13.1 和 ONNX Runtime GPU 1.14.1：

```bash
conda activate rlgpu
python -m pip install -r requirements.txt
```

需要连接 UR5e、D435i 或运行 YOLO 时再安装硬件依赖：

```bash
python -m pip install -r requirements-hardware.txt
```

硬件环境固定使用 `ur-rtde==1.6.5`、`ultralytics==8.0.20` 和
`pyrealsense2==2.55.1.6486`。`constraints-rlgpu.txt` 用于避免安装过程替换现有
Torch/CUDA 组合。网络受限时可以先下载离线包（脚本里用的清华源，关梯子再运行）：

```bash
bash scripts/download_hardware_packages.sh
python -m pip install \
  --no-index \
  --find-links resources/wheels/rlgpu-py37-linux-x86_64 \
  -r requirements-hardware.txt
```

检查显卡、CUDA 和 ONNX Runtime provider：

```bash
conda run --no-capture-output -n rlgpu python scripts/check_rlgpu.py
```
应该输出如下
python=3.7.12
mujoco=2.3.6
onnxruntime=1.14.1
providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
torch_cuda=True

指定 `--provider cuda` 时，如果 CUDA provider 无法创建，程序会直接报错，不会静默退回
CPU。诊断或真机低负载运行可以显式使用 `--provider cpu`。

运行仓库测试：

```bash
conda run --no-capture-output -n rlgpu \
  python -m unittest discover -s tests -v
```

Isaac Gym 的本地路径是：

```text
/home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs
```

路径不同可在导出或预览命令后追加 `--isaac-root /absolute/path/to/IsaacGymEnvs`。

## 仿真预览

### 1. 在 Isaac Gym 中查看原始策略

下面的入口只打开单环境 viewer，不录制参考轨迹：

```bash
./scripts/replay_ur5e_hover_isaacgym.sh
```

该脚本默认加载：

```text
task:  Ur5eRobotiqHoverGripper
model: resources/models/ur5e_hover_gripper/policy.onnx
steps: 300
```

它内部调用 `export_ur5e_isaac_reference.sh --viewer --view-only`。因此预览和后续导出参考
轨迹使用的是同一套环境创建、reset 和 ONNX 推理逻辑；`--view-only` 模式不采集、也不
写入参考轨迹。里面还有很多参数选项例如随机种子、推理步数等可以修改。

### 2. 在 MuJoCo 中查看 ONNX 闭环

```bash
./scripts/run_ur5e_hover_sim2sim.sh --viewer
```

默认输出：

- `resources/trajectories/ur5e_hover_mujoco.npz`
- `resources/trajectories/ur5e_hover_mujoco.csv`

无图形界面时去掉 `--viewer`。只想确认 CPU 链路时使用：

```bash
./scripts/run_ur5e_hover_sim2sim.sh --provider cpu
```

MuJoCo 闭环会读取 hover 专用 scene、策略 metadata 和
`resources/config/ur5e_hover_gripper_sim2sim.yaml`，不能替换成 grasp scene 或旧策略
metadata。

## sim2sim

sim2sim 验证分为两部分：先确认相同 ONNX 在两个引擎中都能闭环运行，再将 Isaac Gym
记录的同一组 action 放入 MuJoCo，检查初始状态、关节顺序、动作映射和动力学分叉。

### 1. 导出 Isaac Gym 参考轨迹

不要加 `--view-only`，否则不会保存文件：

```bash
./scripts/export_ur5e_isaac_reference.sh \
  --isaac-root /home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs \
  --task Ur5eRobotiqHoverGripper \
  --model resources/models/ur5e_hover_gripper/policy.onnx \
  --provider cuda \
  --seed 0 \
  --steps 300 \
  --output resources/trajectories/ur5e_hover_isaac_reference.npz
```

需要边运行边查看时可额外加入 `--viewer`。参考文件包含 observation、action、关节状态、
掌心、指尖和物体状态。

### 2. 在 MuJoCo 中回放同一组 action

```bash
./scripts/run_ur5e_hover_sim2sim.sh \
  --provider cpu \
  --reference resources/trajectories/ur5e_hover_isaac_reference.npz \
  --viewer
```

除了 NPZ 和 CSV，该模式还会写入：

```text
resources/trajectories/ur5e_hover_alignment.json
```

报告用于检查两个引擎的关节顺序、初始状态、动作目标和状态误差。Isaac Gym 与 MuJoCo
使用不同的接触、执行器和约束求解器，长时间轨迹出现差异并不等同于接口错误；应先看
首帧和动作映射，再判断后续误差是否属于动力学分叉。

### 3. checkpoint 变化时重新导出 ONNX

已有 `resources/models/ur5e_hover_gripper/policy.onnx` 可以直接使用。只有更换 checkpoint
时才需要重新导出。导出器要求使用真实的 69 维 observation 轨迹做数值验证，并拒绝
覆盖已有文件：

```bash
/absolute/path/to/Ur5eRobotiqHoverGripperPPO.pth
```
指的是isaacgym里面训练出的.pth文件绝对位置

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/export_ur5e_robotiq_hover_gripper_mlp_onnx.py \
  --checkpoint /absolute/path/to/Ur5eRobotiqHoverGripperPPO.pth \
  --reference resources/trajectories/ur5e_hover_isaac_reference.npz \
  --reference resources/trajectories/ur5e_hover_mujoco.npz \
  --output resources/models/ur5e_hover_gripper/policy_candidate.onnx \
  --metadata resources/models/ur5e_hover_gripper/policy_candidate.meta.yaml
```

候选模型通过 PyTorch/ONNX 数值检查和 sim2sim 后，再更新真机配置(config中对应任务的yaml)中的 `policy.model`、
`policy.metadata` 和 `policy.sha256`。不要只替换 ONNX 而继续使用旧 metadata 或旧哈希。
`policy.sha256`的值需要运行以下命令来获取：
```bash
sha256sum 文件1位置 文件2位置
```
示例在命令行`~/Workspace/Ur5e_deploy/resources/models/ur5e_hover_gripper`位置输入:
```bash
sha256sum policy.onnx policy.pth 
```
输出:
```bash
1fab8538f3dcf4ef78e0dd2a6d2794e6310a4f888bf33a545e440d1c9235e7b6  policy.onnx
c5330c91246aa07d562d371efbf3d03a0793a9e41c0598157541fc3929a12581  policy.pth
```

sim2sim 只验证软件接口和仿真行为，不代表真实工作空间、相机坐标或碰撞风险已经验收。

## 真机部署

真机流程按以下顺序执行：配置与接线 → YOLO 检查 → 腕部相机标定 → 只读环境预览 →
策略 dry-run → 单回合低速执行 → 真机轨迹回放。任何一步失败都不要跳到下一步。

### 1. 核对配置和部署产物

主配置文件是：

```text
resources/config/ur5e_hover_gripper_deploy.yaml
```

运行前至少逐项确认：

- `policy.model`、`policy.metadata` 和 `policy.sha256` 对应同一个已验收策略。
- `observation.model` 和 `observation.model_sha256` 对应当前 hover MJCF。
- `robot.host`、`robot.gripper_port` 和真实初始关节姿态正确。
- `camera.serial` 是腕部 D435i 的序列号。
- `vision.weights`、`vision.target_class` 和真实物体类别一致。
- `camera.calibration` 指向本次安装得到的眼在手上标定文件。
- `workspace_min/max`、`object_position_min/max` 是现场审核后的安全范围。
- `safety.allow_motion` 在调试阶段保持 `false`，完成现场审核后才改为 `true`。

仓库当前 hover 配置中的 `safety.allow_motion` 已是 `true`；如果现场尚未完成验收，应先
改回 `false`。即使配置为 `true`，没有命令行 `--execute` 也不会发送运动命令。

当前策略要求 69 维 observation 和 7 维 action。入口会校验 ONNX I/O、metadata、模型
哈希、关节顺序、MJCF 哈希和标定相机序列号；任一项不一致都会拒绝运行。

### 2. 配置 UR5e、Robotiq 和网络

当前链路为：部署 PC 通过专用以太网连接 UR5e；2F-85 接入 UR 控制柜，由
Robotiq Grippers URCap 提供 `63352` socket。PC 不直接连接夹爪 RS-485，也不使用
External Control URCap。

本实验使用以下网段：

```text
UR5e地址:    192.168.1.10
PC地址:      192.168.1.12
子网掩码: 255.255.255.0
Gateway: 空
```

在 PolyScope 中启用 Remote Control，以及 Dashboard、Primary/Secondary、Real-time 和
RTDE 服务。夹爪应先在示教器中完成扫描、激活、标定和低速开合测试，确认
`0=open`、`255=closed`。

进入到项目根目录(本机的地址):
```bash
cd ~/Workspace/Ur5e_deploy
```
先做网络和端口检查：

```bash
ping -c 3 192.168.1.10
nc -vz 192.168.1.10 29999
nc -vz 192.168.1.10 30002
nc -vz 192.168.1.10 30004
nc -vz 192.168.1.10 63352
```

再执行只读连接测试：

```bash
conda run --no-capture-output -n rlgpu python -c \
  "import rtde_receive; r=rtde_receive.RTDEReceiveInterface('192.168.1.10'); print(r.getActualQ()); r.disconnect()"

conda run --no-capture-output -n rlgpu python -c \
  "import socket; s=socket.create_connection(('192.168.1.10',63352),1); s.sendall(b'GET POS\\n'); print(s.recv(1024).decode().strip()); s.close()"
```
正常会分别返回:
```bash
[-1.5707481543170374, -1.5707958501628418, -1.5708074569702148, -1.5707605642131348, 1.5707731246948242, 1.1920928955078125e-05]
```
```bash
POS 105
```
如果没有返回值就需要检查IP是否正确、ur5e是否激活、远程模式是否开启。

### 3. 准备并检查 YOLO

已有权重可先对单张图片或目录内全部图片做离线检查：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/predict_yolo_image.py path/to/image_or_directory \
  --model resources/models/vision/object_yolo.pt \
  --device cpu
```

标注结果默认写入 `resources/predictions/yolo/`。
连接腕部相机后检查实时检测框、目标类别、bbox 中心和对齐深度：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/inspect_yolo_realsense.py \
  --config resources/config/ur5e_hover_gripper_deploy.yaml \
  --view
```
正常情况下会打开一个窗口实时显示识别状态并返回坐标。

按 `q` 或 `Esc` 退出。若需要重新训练，及记得修改`data.yaml`中数据的地址和数据包含的类，数据集应使用标准 YOLO Detection 目录结构，然后
运行：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/train_yolo_object.py \
  --data /absolute/path/to/object_yolo_dataset/data.yaml \
  --device 0
```

确认验证结果后再加入 `--publish`，将 `best.pt` 发布为
`resources/models/vision/object_yolo.pt`。部署配置中的 `vision.target_class` 必须存在于
权重的类别名称中。

### 4. 标定腕部相机

hover 部署只使用腕部 D435i。标定采集参数目前保存在
`resources/config/ur5e_deploy.yaml`；先把其中 `camera.serial` 设置成与 hover 配置相同的
腕部相机序列号，再运行：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/calibrate_eye_on_hand.py \
  --config resources/config/ur5e_deploy.yaml \
  --output resources/calibration/eye_on_hand.yaml
```

标定板为 DFVision `Q12-240-15`，OpenCV 内角点为 `11 × 8`，方格边长为 `15 mm`。
使用示教器手动改变 TCP 位置和三个旋转轴方向；默认采集 20 个姿态，至少保留 12 个
内点。TCP 平移跨度至少 `0.10 m`、旋转跨度至少 `30°`。采样时保持同一物理角作为画面
左上角，避免普通棋盘格的 180° 朝向歧义。

相机安装、active TCP 或夹具发生变化后必须重新标定。最终
`resources/calibration/eye_on_hand.yaml` 中的相机序列号必须与 hover 部署配置一致。

### 5. 只读检查真实目标和安全边界

以下入口只创建 RTDE receive 和 Robotiq `GET POS`，不会创建 RTDE control、激活夹爪或
发送运动命令。它会把相机实时检测到的物体位置投影到 MuJoCo，并显示物体范围和掌心工作
空间：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/view_ur5e_hover_camera_object_mujoco.py
```

也可以不连接硬件，直接设置物体坐标（相对于ur5e基座）：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/view_ur5e_hover_camera_object_mujoco.py \
  --object-base -0.10 -0.49 0.15
```

确认目标位于红色物体边界内、机械臂初始姿态正确，并根据真实工位收紧配置中的工作空间
和物体范围。MJCF 的关节极限不能代替现场碰撞检查。

### 6. 运行真机 dry-run

不加 `--execute` 时，策略入口保持只读：不创建 RTDE control、不激活夹爪、不发送机械臂
或夹爪命令，但仍会读取真实状态、运行 YOLO、构造 69 维 observation、推理 action 并
执行全部安全检查：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/run_ur5e_hover_gripper_policy.py \
  --provider cpu \
  --steps 300
```

机械臂必须由人工放到配置中的 `robot.initial_joint_position`，夹爪应为空且处于张开位置。
程序会等待稳定的目标锁定，并把结果保存到：

```text
resources/trajectories/ur5e_hover_gripper_real/
```

每回合包含 NPZ 全量轨迹、JSON 摘要和物体状态 CSV。只有 dry-run 完整通过，且检测坐标、
observation、action、关节目标和安全边界均合理，才进入实机运动。

### 7. 执行单个真机回合

真实运动需要同时满足：

1. `resources/config/ur5e_hover_gripper_deploy.yaml` 中
   `safety.allow_motion: true`；
2. 命令行显式加入 `--execute`；
3. 启动时人工输入大写 `ARM`。

首次执行只运行一个 300 步回合：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/run_ur5e_hover_gripper_policy.py \
  --provider cpu \
  --steps 300 \
  --execute
```

执行期间持续检查急停/保护状态、RTDE 状态新鲜度、关节速度、单步增量、跟踪误差、掌心
工作空间、目标范围和相机目标状态。异常或 `Ctrl-C` 会停止伺服；中止回合不会自动回到
初始姿态，应由现场人员监督恢复。

完整且无异常的回合结束后，程序才会询问是否输入 `RETURN`。只有确认夹爪为空、回程路径
无障碍后才允许回初始位姿；直接回车则保持当前位置。

单回合和自动回位均完成验收后，才考虑连续模式：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/run_ur5e_hover_gripper_policy.py \
  --provider cpu \
  --steps 300 \
  --execute \
  --continuous
```

连续模式首次仍需输入 `ARM`，每回合安全返回后由操作员确认是否开始下一回合。运行期间
策略、metadata、MJCF 或标定文件发生变化时，程序会终止连续模式。

### 8. 回放和复核真机轨迹

默认回放最新的非空真机轨迹，不连接机器人：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/replay_ur5e_hover_real_mujoco.py
```

指定轨迹或调整回放速度：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/replay_ur5e_hover_real_mujoco.py \
  --trajectory resources/trajectories/ur5e_hover_gripper_real/episode_XXXX.npz \
  --speed 0.5
```

回放会复现机械臂、夹爪和目标状态，并报告记录掌心位置与 MuJoCo 正运动学之间的误差。
每次实机执行后都应同时检查 JSON 中的 `error`、`success`、完成步数和回位结果，以及
CSV 中的目标位置、速度、状态年龄和位移。
