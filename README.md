# Robot policy sim2sim and deployment

本仓库按验收门分三阶段推进：Allegro-Rokae MuJoCo sim2sim、UR5e+Robotiq
sim2sim、UR5e 实机部署。当前只实现并验收第一阶段；后两阶段等待对应资产、ONNX
和策略 metadata，不会猜测观测或动作接口。

## Project Structure

- `mujoco_sim`：MuJoCo 模型构建、机器人 adapter 和 episode runner。
- `onnx_deploy`：带 LSTM/GRU 状态的 ONNX Runtime runner。
- `resources/assets`：机器人源 URDF、mesh 和生成的 MJCF scene。
- `resources/models`：ONNX 与接口 metadata。
- `ur5e_comm`：后续 UR RTDE 与 Robotiq TCP 实机通信。
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

UR5e+Robotiq 的 MJCF 与 mesh 资产门已经具备。策略 sim2sim 开始前仍需提供 ONNX、
观测/动作 metadata、归一化、控制频率和循环状态定义。该阶段通过后才实现 UR RTDE
+ Robotiq TCP；实机接口将默认 dry-run，只有显式 arm 后才允许发送控制命令。
