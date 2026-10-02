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
  --provider cpu \
  --steps 600 \
  --viewer
```

该入口默认使用已筛出的最高成功率权重
`successful_best_seed0/rank01_maxsucc13/policy.onnx`。若要与 Isaac Gym 比较同一物体、
同一目标和同一初始机器人状态，先采集单环境参考：

```bash
scripts/export_rokae_isaac_reference.sh --provider cpu --seed 0 --steps 600
```

然后回放完全相同的动作并生成逐关节、逐指尖、物体和事件对齐报告：

```bash
conda run --no-capture-output -n rlgpu \
  python -m mujoco_sim.rokae_alignment \
  --reference resources/trajectories/rokae_successful_rank01/isaac_seed0_reference.npz
```

也可从该参考首帧启动 ONNX 闭环并在 MuJoCo 中观察分叉过程：

```bash
scripts/run_rokae_sim2sim.sh --provider cpu --steps 600 --viewer \
  --playback-speed 0.25 \
  --reference-initial resources/trajectories/rokae_successful_rank01/isaac_seed0_reference.npz
```

`--playback-speed 0.25` 只把 Viewer 放慢到四分之一实时速度，不改变策略周期、物理
timestep 或轨迹结果。

当前对齐已经消除了关节顺序、动作限位、掌心偏置/速度、物体质量、摩擦、初始几何和
自碰撞配置错误。首帧掌心/指尖位置误差低于 `0.04 mm`，动作目标 RMSE 为
`2.8e-7 rad`。剩余差异属于动力学模型：Isaac 实际导入的 24 个机器人刚体中有 7 个
惯量不满足刚体三角不等式，而 MuJoCo 必须自动平衡为合法惯量；报告会明确列出此计数和
惯量误差。因此现有 PhysX checkpoint 不能通过参数复制获得严格的 MuJoCo 成功率。
如需两引擎都稳定成功，应先修正训练 URDF 惯量，并在包含 PhysX/MuJoCo 参数随机化的
任务上重新训练或微调；不要用增大摩擦或隐藏状态回放伪装成 sim2sim 成功。

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

### UR5e grasp 烟雾训练与 sim2sim

烟雾任务使用固定窄桌、`0.04 m` 方块（`567 kg/m³`）和物体上方 `0.12 m` 虚拟目标，
不创建桶。训练前先校验 URDF、PhysX 实际导入值和 MuJoCo 编译值：

```bash
./scripts/train_ur5e_grasp_smoke.sh
```

该入口仅在惯量审计通过后，以 seed 0、256 环境、horizon 16、LSTM 768 运行 30 epoch，
输出位于 IsaacGymEnvs 的 `train_dir/ur5e_grasp_smoke_seed0/`。当前稳定 checkpoint 为
`ur5e_grasp_smoke_epoch30.pth`；它能产生非恒定机械臂/夹爪动作，但没有抓取成功，故导出
metadata 明确标记为 `smoke_only`。导出器严格恢复 normalization/LSTM 参数，并比较连续
5 步 PyTorch/ONNX 状态递推：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/export_ur5e_policy_onnx.py \
  --checkpoint /home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs/train_dir/ur5e_grasp_smoke_seed0/ur5e_grasp_smoke_epoch30.pth \
  --provider cpu
```

采集 Isaac 参考、生成 MuJoCo grasp scene，并回放同一动作：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/export_ur5e_isaac_reference.py \
  --isaac-root /home/liang/Workspace/Isaacgym_cjlu/IsaacGymEnvs \
  --seed 0 --output resources/trajectories/ur5e_grasp_isaac_reference.npz
conda run --no-capture-output -n rlgpu python -m mujoco_sim.build_ur5e_grasp_scene
conda run --no-capture-output -n rlgpu \
  python -m mujoco_sim.run_ur5e_sim2sim \
  --provider cpu \
  --reference resources/trajectories/ur5e_grasp_isaac_reference.npz
```

不传 `--reference` 时运行 ONNX 闭环；加 `--viewer` 可实时显示。当前同动作结果的初始
关节/掌心/指尖及动作目标映射通过阈值，但失败策略持续饱和后，两引擎的接触、限位和
驱动求解分叉，完整接触前关节 RMSE 不通过，因此报告结论是“链路不一致、任务失败”。
这份结果用于暴露差异，不得视为抓取策略或 sim2real 性能验收。

独立加载、刚性安装、执行器范围和 1000-step 动力学测试：

```bash
conda run --no-capture-output -n rlgpu \
  python -m unittest tests.test_ur5e_robotiq_assets -v
```

## Next gates

UR5e+Robotiq 的 MJCF、grasp smoke ONNX 和实机软件闭环已经具备，但当前策略只用于
联调。首次带电运动前仍必须完成下方外部验收项和完整 dry-run。

## UR5e 实机部署

### 策略接口

参考任务实际使用 12 个物理关节状态（6 个 UR5e + 6 个 Robotiq linkage）、2 个
指尖和 1 个物体中心 keypoint，因此 observation 是 69 维，action 是 7 维。完整切片
记录在 `resources/models/ur5e_robotiq/policy.meta.yaml`。机械臂的关节位置、速度、TCP、
TCP 速度、电流和目标力矩均从 RTDE 读取并写入每回合轨迹；训练 observation 所需的
掌心、指尖位姿和速度通过仓库内同一套 UR5e+2F85 MJCF 正运动学重建。

部署使用两台 D435i：eye-on-hand 相机在每回合开始时定位目标，固定 eye-on-base 相机
交叉确认后在整回合持续跟踪。bbox 中心邻域的深度中位数是物体表面深度；程序增加
`0.01 m` 后沿同一像素射线反投影，再转换到 UR base。固定相机按配置帧率更新位置，
策略以滤波位置和线速度更新相对掌心/目标、lifted 和 reward；物体四元数保持配置值，
角速度为零。短时丢检使用有界匀速预测，状态超过 `0.15 s` 即在下一条控制命令前停止。

### 1. 安装实机依赖

```bash
conda activate rlgpu
python -m pip install -r requirements-hardware.txt
```

硬件依赖锁定为 `ur-rtde==1.6.5`、`ultralytics==8.0.20` 和
`pyrealsense2==2.55.1.6486`。`constraints-rlgpu.txt` 同时保护现有 Torch 1.8.1、
Torchvision 0.9.1、NumPy 1.21.6 和 CUDA 11 组合，安装后可用部署 preflight 核对
实际导入和版本。

部署机使用 Ubuntu HWE low-latency 软实时内核，并配套 Canonical 预编译、签名的
NVIDIA 模块；不要在这台 CUDA 主机上强制绕过 NVIDIA 的 PREEMPT_RT 检查：

```bash
sudo apt-get install -y \
  linux-lowlatency-hwe-22.04 \
  linux-modules-nvidia-610-lowlatency-hwe-22.04 \
  rt-tests
```

当前验收版本为 `6.8.0-142-lowlatency`、NVIDIA `610.57.04`。`realtime` 组的
`RLIMIT_RTPRIO` 必须至少为 90；部署配置将 RTDE receive、RTDE control 和 Python
500 Hz servo feeder 分别设为 FIFO `90/85/80`。用 `cyclictest -p90 -t1 -i2000
-D30s -m -q` 验证 2 ms 周期，且必须在 YOLO/CUDA 并发负载下复测。

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

### 2. 配置控制柜侧 UR5e 与 Robotiq 驱动

本项目采用以下固定链路：PC 通过专用以太网直接运行 `ur-rtde`；2F-85 的外部线缆接入
UR 控制柜，由控制柜内的 Robotiq 原装 USB-RS485 转换器和 **Robotiq Grippers URCap**
驱动；PC 上的程序只连接 `192.168.1.10:63352`。因此不要在 PC 安装 `pyserial`，不要把
夹爪接到 PC 的 USB，也不要改成本机 Modbus RTU 后端。

在控制柜断电并执行现场上锁/挂牌后，按夹爪和控制柜对应版本的官方接线图完成接线：

- 夹爪外部线缆的电源线接控制柜 `24 V/0 V`，RS-485 信号线和屏蔽线接 Robotiq 原装
  USB-RS485 转换器，再将转换器插入控制柜 USB；端子定义、极性、屏蔽接地及控制柜
  24 V 余量必须以实物手册和线缆料号为准。
- 不从 PC USB 给夹爪供电，不并联第二个 RS-485 终端电阻，也不要同时接腕部 Tool I/O。
- 上电前用万用表核对极性、电压和无短路；安装夹爪、coupling、两台相机及安装件后，
  在 UR installation 中填写总 payload、质心和 active TCP。相机标定完成后不得修改 TCP。

在 PolyScope 5 中安装与控制器版本兼容的 **Robotiq Grippers URCap** 并重启控制器；在
Installation 的 Robotiq 页面依次执行扫描、激活和重新标定，先用示教器低速开合确认
`0=open、255=closed`。这条控制柜接线需要 Robotiq URCap，但不需要 Robotiq Wrist
Connection URCap。本项目的 `ur-rtde` 使用默认脚本上传方式，也不安装或选择 Universal
Robots External Control URCap。

在 PolyScope 中启用 Remote Control，并在 Security/Services 中启用 Dashboard、
Primary/Secondary、Real-time 和 RTDE 服务。设置专用静态网络：

- UR5e：`192.168.1.10/24`，网关和 DNS 留空。
- 部署 PC 网口：`192.168.1.20/24`，网关和 DNS 留空；不要与 Wi-Fi 或其他网口使用重叠网段。
- 连接前保持机器人在 Local 模式完成示教器设置；运行实机程序时切换到 Remote Control。

控制柜配置完成后，先做不发送运动命令的连通性检查：

```bash
ping -c 3 192.168.1.10
nc -vz 192.168.1.10 29999
nc -vz 192.168.1.10 30002
nc -vz 192.168.1.10 30004
nc -vz 192.168.1.10 63352
```

下面两项分别只读取机械臂关节和夹爪位置。夹爪应返回 `POS 0..255`，不得返回空响应：

```bash
conda run --no-capture-output -n rlgpu python -c \
  "import rtde_receive; r=rtde_receive.RTDEReceiveInterface('192.168.1.10'); print(r.getActualQ()); r.disconnect()"

conda run --no-capture-output -n rlgpu python -c \
  "import socket; s=socket.create_connection(('192.168.1.10',63352),1); s.sendall(b'GET POS\\n'); print(s.recv(1024).decode().strip()); s.close()"
```

现场按“24 V/接线检查 → URCap 示教器低速开合 → `63352` 只读 → RTDE 只读 → 完整
dry-run → 低速单回合运动”的顺序验收。前一项未通过时不要进入下一项。

### 3. 训练物体检测 YOLO

在 X-AnyLabeling 中只使用矩形框并导出为 YOLO Detection 格式；可以包含多个类别。
需要作为强化学习初始物体位置的类别必须出现在 `data.yaml` 的 `names` 中，并与
`resources/config/ur5e_deploy.yaml` 的 `vision.target_class` 完全一致。训练脚本要求
标准目录，且每张图片都要有合法的归一化标签：

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

`data.yaml` 可参考 `resources/config/data.yaml`；当前示例包含多个类别，类别 ID 必须从
0 开始连续排列。脚本会先校验类别、目录和每一个 bbox，再训练、
用 `best.pt` 验证并对第一张验证图片做 smoke
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

训练或发布后可在不连接 UR5e 的情况下检查 D435i 多类别识别、bbox 中心、对齐深度和
相机坐标。打开实时彩色流时会绘制所有类别的检测框、中心十字、置信度和中心深度；
`TARGET` 是正式部署会选中的目标（目标类别中置信度最高的检测框）。按 `q`、`Esc` 或
关闭窗口退出：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/inspect_yolo_realsense.py --view
```

不加 `--view` 时默认只处理并打印一帧。可用 `--weights`、`--target-class`、
`--confidence`、`--device`、`--frames` 和 `--print-every` 临时覆盖行为。
部署环境固定为 Python 3.7 / Ultralytics 8.0.20；对于新版 Ultralytics 8.x 训练、但仍由
旧版已有 YOLOv8 层组成的 `.pt`，加载器会注册只读模块路径兼容别名，并关闭运行时自动
安装。若权重包含旧版不存在的新层，程序会明确要求在训练环境导出 ONNX，而不会尝试
联网修改部署环境。

### 4. 导出选定 checkpoint

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

### 5. 配置与双相机标定

编辑 `resources/config/ur5e_deploy.yaml` 中的机器人 IP、目标桶中心、安全边界、YOLO 参数，
并填写两台 D435i 各自且不同的序列号。腕部相机刚性安装在当前 UR TCP 上；先将 DFVision
`Q12-240-15` 棋盘固定在工作空间，用示教器手动改变腕部相机位置与朝向并采集：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/calibrate_eye_on_hand.py
```

默认采集 20 个姿态，至少保留 12 个内点。姿态应覆盖三个旋转轴，TCP 平移跨度至少
`0.10 m`、旋转跨度至少 `30°`；不要只在一个平面平移。脚本拒绝运动中、重投影误差过大
或与已有样本过近的样本，并在 `resources/calibration/sessions/` 保存角点图、原始变换和
残差。输出 `eye_on_hand.yaml` 中的 `T_tcp_camera` 将相机光学坐标系转换到标定时的 UR TCP。
修改示教器的 active TCP 或相机安装后必须重新标定。

随后固定第二台 D435i。将同一棋盘刚性安装到 TCP，固定相机保持不动，用示教器手动改变
TCP 姿态并采集；棋盘相对 TCP 的安装偏置无需测量，脚本会与 `T_base_camera` 同时求解：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/calibrate_eye_on_base.py
```

脚本同样不会创建 RTDE control 或发送运动命令。输出的 `eye_on_base.yaml` 保存
`T_base_camera`，原始样本和闭环残差位于 `resources/calibration/eye_on_base_sessions/`。
固定相机、棋盘安装或 active TCP 在采样期间发生变化都会使结果无效。

普通棋盘格存在 180° 朝向歧义；采样时保持同一物理角作为图像左上方向，不要让棋盘在
画面中翻转 180°。若无法保证，应改用带 ID 的 ChArUco 标定板。

每回合先用 `T_base_tcp · T_tcp_camera` 得到腕部定位，再用固定 `T_base_camera` 定位；两者
相差不超过 `0.05 m` 才会启动策略。在线跟踪使用最近预测位置关联同类目标，默认门限
`0.25 m`；位置/速度 EMA、预测时长和最大状态年龄均可在 `tracking` 下现场调节。

### 6. 只读 dry-run

先运行离线 preflight；它会逐项列出尚缺的 ONNX、YOLO、标定文件和 Python 依赖：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/check_ur5e_deployment.py
```

默认模式只连接 RTDE receive 和 Robotiq `GET POS`，不创建 RTDE control、不激活夹爪、
不发送停止或运动命令。每回合开始前只读核对机械臂已由人工置于训练 home、夹爪
`POS<=5`，不满足就终止：

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/run_ur5e_policy.py --provider cpu --episodes 1 --steps 60
```

确认检测坐标、69 维 observation、7 维 action、循环状态和轨迹文件均合理。普通策略
每回合结束后记录命中 `y/x`；`smoke_only` 策略只记录 `passed/aborted` 联调结果，不计入
抓取成功率。结果及检测信息追加到
`resources/trajectories/ur5e_real/episodes.jsonl`，全维数据保存在对应 NPZ。

策略目标按训练频率 60 Hz 更新；实机层用独立的 500 Hz `servoJ` 线程持续保持最新关节
目标。Robotiq 绝对位置目标限频到 20 Hz，避免其 URCap socket 往返阻塞机械臂伺服。

### 7. 解锁实机运动

普通策略发送控制需要同时满足三项：配置中设置 `safety.allow_motion: true`、命令行加入
`--execute`、启动时人工输入大写 `ARM`。`smoke_only` 还必须加入
`--allow-smoke-policy`，并固定运行 600 步。运行期间会检查保护/急停、状态新鲜度、关节
速度、硬限位、单步增量、跟踪误差和掌心工作空间；异常或 `Ctrl-C` 会进入停止清理。
每个回合检测前还会要求输入大写 `HOME`，随后先张开空夹爪，再以配置的低速 `moveJ`
回到训练初始关节姿态；到位误差和静止状态确认通过后才采集相机帧。

```bash
conda run --no-capture-output -n rlgpu \
  python scripts/run_ur5e_policy.py --execute --allow-smoke-policy --episodes 1
```

smoke 档会强制机械臂动作乘 `0.05`、单步不超过 `0.008 rad`、实测速度不超过
`0.5 rad/s`、目标误差不超过 `0.06 rad`，夹爪使用 `speed=32/force=20`。这些限制不会
修改生产默认值；首次执行仍须使用单回合、轻质方块、隔离工作区并保持急停可触达。

### 尚需现场提供/验收

- 当前 `smoke_only` ONNX 已生成，但它不能替代后续正式训练和独立验收的生产策略。
- YOLO 权重、目标 class、两台 RealSense 序列号，以及现场生成的 `eye_on_hand.yaml` 和
  `eye_on_base.yaml`。
- 真实桶中心、物体 scale/初始姿态约定和保守的关节/笛卡尔安全边界。
- 控制柜侧外部 RS-485 接线与 24 V 供电已验收；Robotiq Grippers URCap 已激活并在
  `63352` 提供服务。PC 不安装串口驱动或 Modbus 后端。
- UR 控制器启用 Remote Control 及 Dashboard、Primary/Secondary、Real-time、RTDE 服务；
  不需要 External Control URCap。专用网络按 `192.168.1.10/24 ↔ 192.168.1.20/24` 验收。
- 当前 smoke ONNX 的 Isaac reference 和 MuJoCo 数值报告已生成且显示长时动力学不一致；
  现场仍需先做只读 dry-run，再做受限实机联调。正式策略必须重新采集并单独验收。
