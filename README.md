# Robot policy sim2sim and deployment

本仓库包含 Allegro-Rokae MuJoCo sim2sim、UR5e+Robotiq 资产，以及默认只读的
UR5e 实机部署闭环。实机 observation/action 已按本机
`Isaacgym_cjlu/.../ur5e_robotiq_base.py` 对齐为 69/7 维；真实策略 ONNX、YOLO
权重和本工作站标定结果仍是必须由部署者提供的外部产物。

## Project Structure

- `mujoco_sim`：MuJoCo 模型构建、机器人 adapter 和 episode runner。
- `onnx_deploy`：带 LSTM/GRU 状态的 ONNX Runtime runner。
- `resources/assets`：机器人源 URDF、mesh 和生成的 MJCF scene。
- `resources/models`：ONNX 与接口 metadata。
- `ur5e_comm`：相机标定、YOLO 初始定位、UR RTDE、Robotiq TCP、状态投影和安全闭环。
- `tests`：模型、观测、动作、循环状态和闭环测试。

## Environment

```bash
conda activate rlgpu
python -m pip install -r requirements.txt
```

默认使用 `CUDAExecutionProvider`；无 CUDA 的机器可在运行命令中显式使用
`--provider cpu`。本项目复用 `rlgpu` 中已验证的 ONNX 1.13.1、
ONNX Runtime GPU 1.14.1 和 CUDA 11 运行库；CUDA provider 无法创建时 runner
会报错退出，不会静默回退到 CPU。

运行 CUDA 前先检查驱动与环境（仅 provider 出现在列表中并不代表 GPU 可用）：

```bash
nvidia-smi
conda run --no-capture-output -n rlgpu python scripts/check_rlgpu.py
```

## Allegro-Rokae sim2sim

机器人 MJCF 由训练时使用的 URDF 生成。构建器会启用惯量修复、保留 23 个训练关节，
并添加 position actuator、palm/fingertip sites、table、object 和 bucket：

```bash
python -m mujoco_sim.build_rokae_mjcf
```

无窗口运行一个 600-step episode，并保存 NPZ 与 CSV 轨迹：

```bash
scripts/run_rokae_sim2sim.sh \
  --provider cuda \
  --steps 600
```

仅在无 GPU 时进行额外 CPU 诊断：

```bash
scripts/run_rokae_sim2sim.sh --provider cpu --steps 600
```

打开 MuJoCo viewer：

```bash
scripts/run_rokae_sim2sim.sh \
  --provider cuda \
  --steps 600 \
  --viewer
```

策略周期为 0.01667 秒，MuJoCo timestep 为 0.002 秒。runner 通过目标仿真时间
自动交替执行 physics substeps，避免把 0.01667 错误截断成固定 8 个 timestep。

## Validation

```bash
conda run --no-capture-output -n rlgpu \
  python -m unittest discover -s tests -v
```

测试覆盖：23 关节/执行器顺序、99 维 observation、四元数顺序、动作映射与限位、
LSTM reset/回传，以及 600-step headless 闭环和轨迹有限性。

## UR5e + Robotiq 2F85 assets

机器人资源来自本机仓库
`/home/liang/Workspace/MuJoCo-UR5e-with-Robotiq-2F85-and-3F-Grippers`。
当前仓库保留独立的 UR5e/2F85 组件、mesh 和各自许可证；生成后的模型只使用仓库内
相对路径，不依赖源仓库。夹爪使用 `robotiq_2f_85_gripper_visualization` 的 Xacro、
惯量和 visual/collision mesh；不再使用旧 `components/robotiq_2f85` 模型。由于
MuJoCo 2.3.6 不能读取 DAE，首次使用或官方 mesh 更新后先生成等价 OBJ：

```bash
conda run --no-capture-output -n rlgpu \
  python -m mujoco_sim.convert_robotiq_visualization_meshes
```

然后重新生成静态合体模型：

```bash
conda run --no-capture-output -n rlgpu \
  python -m mujoco_sim.build_ur5e_robotiq_2f85_mjcf
```

合体模型位于
`resources/assets/robots/ur5e_robotiq_2f85/ur5e_robotiq_2f85.xml`。它将 2F85
固定安装到 UR5e 法兰，不使用 free joint 或运行时 weld。模型包含 12 个关节和 7 个
执行器：前六个 actuator 控制 UR5e，`fingers_actuator` 使用 `0–255` 控制 2F85。

关节顺序固定为：

```text
shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
wrist_1_joint, wrist_2_joint, wrist_3_joint,
finger_joint, left_inner_finger_joint, left_inner_knuckle_joint,
right_outer_knuckle_joint, right_inner_finger_joint,
right_inner_knuckle_joint
```

2F85 严格按 visualization Xacro 拼装：`outer_finger` 固定在 `outer_knuckle`，
`inner_finger` 以 revolute 挂在 `outer_finger`，pad 固定在 `inner_finger`；
`inner_knuckle` 是 base 的独立 revolute 分支。`finger_joint` 是唯一主动关节，左右
outer knuckle 与 inner knuckle 按 `+1` mimic，两个 inner finger 按 `-1` mimic。
`fingers_actuator` 通过带符号的 `finger_coupling` fixed tendon 向六个 mimic joint
直接分配驱动力，控制范围保持 `0–255`；五条 equality 同时约束角度关系。

夹爪资源包在 `package.xml` 中声明 BSD 许可；模型来源与维护者信息保留在工作区内的
`components/robotiq_2f_85_gripper_visualization/`。

独立加载、刚性安装、执行器范围和 1000-step 动力学测试：

```bash
conda run --no-capture-output -n rlgpu \
  python -m unittest tests.test_ur5e_robotiq_assets -v
```

## Next gates

UR5e+Robotiq 的 MJCF 与 mesh 资产门已经具备。实机软件闭环也已经实现，但首次
带电运动前仍必须完成下方“实机部署”中的外部验收项，尤其是匹配 checkpoint 的
sim2sim 和低速小步测试。

## UR5e 实机部署

### 策略接口

参考任务实际使用 12 个物理关节状态（6 个 UR5e + 6 个 Robotiq linkage）、2 个
指尖和 1 个物体中心 keypoint，因此 observation 是 69 维，action 是 7 维。完整切片
记录在 `resources/models/ur5e_robotiq/policy.meta.yaml`。机械臂的关节位置、速度、TCP、
TCP 速度、电流和目标力矩均从 RTDE 读取并写入每回合轨迹；训练 observation 所需的
掌心、指尖位姿和速度通过仓库内同一套 UR5e+2F85 MJCF 正运动学重建。

按需求，YOLO+深度相机只在每回合开始时取得一次物体中心。之后物体位置和姿态固定，
物体线速度、角速度、lifted、在线 reward 均填 0；LSTM 状态仍逐步传递。这个行为与
训练时可持续读取仿真物体状态不同，属于明确的 sim-to-real 分布偏移，必须在
sim2sim/离线回放中单独验证成功率。

### 1. 安装实机依赖

```bash
conda activate rlgpu
python -m pip install -r requirements-hardware.txt
```

硬件依赖锁定为 `ur-rtde==1.6.5`、`ultralytics==8.0.20` 和
`pyrealsense2==2.55.1.6486`。`constraints-rlgpu.txt` 同时保护现有 Torch 1.8.1、
Torchvision 0.9.1、NumPy 1.21.6 和 CUDA 11 组合，安装后可用部署 preflight 核对
实际导入和版本。

网络较慢时，可先从清华 PyPI 镜像单独下载缺少的 wheel/源码包，再离线安装。下载清单
在 `requirements-hardware-download.txt`，不包含已有的 Torch/CUDA 大包：

```bash
conda activate rlgpu
bash scripts/download_hardware_packages.sh
python -m pip install \
  --no-index \
  --find-links resources/wheels/rlgpu-py37-linux-x86_64 \
  -r requirements-hardware.txt
```

如需更换镜像，可设置 `PYPI_MIRROR_URL`；wheel 目录已加入 `.gitignore`，可复制到
相同 Python 3.7 / Linux x86_64 的离线部署机使用。脚本按清单逐包下载并默认使用 6 个
并发连接；网络有限时可通过 `DOWNLOAD_JOBS=2` 调低。

### 2. 训练物体检测 YOLO

在 X-AnyLabeling 中只使用矩形框，类别名固定为 `object`，导出为 YOLO Detection
格式。训练脚本要求标准目录，且每张图片都要有合法的归一化标签：

```text
object_yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

`data.yaml` 可参考 `resources/config/yolo_object_data.example.yaml`。脚本会先校验
类别、目录和每一个 bbox，再训练、用 `best.pt` 验证并对第一张验证图片做 smoke
推理。脚本会从系统 DejaVu/Liberation 字体创建本地训练缓存，避免 Ultralytics 8.0.20
首次运行时联网下载 `Arial.ttf`。默认不会覆盖部署权重：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/train_yolo_object.py \
  --data /absolute/path/to/object_yolo_dataset/data.yaml \
  --device 0
```

默认 `--model yolov8n.pt` 会在首次使用时下载官方 COCO 预训练权重。离线训练时先自行
准备该文件，再显式传入 `--model /absolute/path/to/yolov8n.pt`；也可传 Ultralytics
安装目录中的 `yolov8n.yaml` 从随机初始化开始训练，但通常不建议用于实际数据。

查看运行目录中的 `results.csv`、验证图和 `training_metadata.yaml`。确认指标满足现场
要求后，显式发布最佳权重：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/train_yolo_object.py \
  --data /absolute/path/to/object_yolo_dataset/data.yaml \
  --device 0 \
  --publish
```

发布得到 `resources/models/vision/object_yolo.pt` 与 `object_yolo.meta.yaml`。续训时仍
提供同一个 `--data`，并指定上次运行的未裁剪 checkpoint：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/train_yolo_object.py \
  --data /absolute/path/to/object_yolo_dataset/data.yaml \
  --resume /absolute/path/to/run/weights/last.pt
```

### 3. 导出选定 checkpoint

仓库不会从训练目录中的大量 checkpoint 擅自选择权重。先按仿真评估结果选定一个，
再严格恢复完整 rl-games 模型（含输入归一化统计）并导出、数值校验：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/export_ur5e_policy_onnx.py \
  --checkpoint /absolute/path/to/selected_Ur5eRobotiq_checkpoint.pth \
  --provider cuda
```

输出为 `resources/models/ur5e_robotiq/policy.onnx`，同时更新 checkpoint/ONNX 哈希和
验证误差。部署入口还会再次校验 ONNX 的 69/7 维接口与关节顺序。

### 4. 配置与 eye-on-base 标定

编辑 `resources/config/ur5e_deploy.yaml` 中的机器人 IP、目标桶中心、安全关节/工作空间、
YOLO 权重和目标类别。测量二维码中心在 UR base 下的位姿，写入
`calibration.base_to_qr`，再采集标定：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/calibrate_eye_on_base.py
```

标定文件保存的 `T_base_camera` 将相机光学坐标系中的米制点变换到 UR base 坐标系。
运行时使用 YOLO bbox 中心附近有效深度的中位数，并结合对齐后的彩色相机内参完成反投影。

### 5. 只读 dry-run

先运行离线 preflight；它会逐项列出尚缺的 ONNX、YOLO、标定文件和 Python 依赖：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/check_ur5e_deployment.py
```

默认模式只连接 RTDE receive 和 Robotiq `GET POS`，不创建 RTDE control、不激活夹爪、
不发送停止或运动命令：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/run_ur5e_policy.py --provider cpu --episodes 1 --steps 60
```

确认检测坐标、69 维 observation、7 维 action、循环状态和轨迹文件均合理。每回合结束
后，命中输入 `y`，否则输入 `x`；结果及检测信息追加到
`resources/trajectories/ur5e_real/episodes.jsonl`，全维数据保存在对应 NPZ。

策略目标按训练频率 60 Hz 更新；实机层用独立的 500 Hz `servoJ` 线程持续保持最新关节
目标。Robotiq 绝对位置目标限频到 20 Hz，避免其 URCap socket 往返阻塞机械臂伺服。

### 6. 解锁实机运动

实机发送控制需要同时满足三项：配置中设置 `safety.allow_motion: true`、命令行加入
`--execute`、启动时人工输入大写 `ARM`。运行期间会检查保护/急停、状态新鲜度、关节
速度、硬限位、单步增量、跟踪误差和掌心工作空间；异常或 `Ctrl-C` 会进入停止清理。

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/run_ur5e_policy.py --execute --episodes 1
```

首次执行必须降低 `max_arm_step` 和夹爪速度、使用单回合、保持示教器急停可触达。

### 尚需现场提供/验收

- 经仿真选择并导出的 UR5e 69→7 ONNX（当前仓库没有该文件）。
- YOLO 权重、目标 class、RealSense 序列号，以及现场生成的 `eye_on_base.yaml`。
- 真实桶中心、物体 scale/初始姿态约定和保守的关节/笛卡尔安全边界。
- UR 控制器启用 External Control/Remote 模式、63352 端口上的 Robotiq URCap 服务。
- 使用同一 ONNX 的 UR5e sim2sim 轨迹对齐；通过后再做只读 dry-run 和低速实机测试。
