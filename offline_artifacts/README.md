# 未上传资源包

该目录用于单独保存被 `.gitignore` 排除、不会随 Git 仓库上传的运行资源。压缩包只包含
Git 判定为 ignored 的文件，保留相对于仓库根目录的原始路径；已跟踪源码、metadata、
MJCF 以及 `__pycache__`、`.pyc` 等缓存不会重复打包。

## 包含内容

生成的压缩包位于 `archives/`，按原始一级资源目录拆分：

| 压缩包 | 恢复目录 | 内容 |
| --- | --- | --- |
| `resources_models_ignored.tar.gz` | `resources/models/` | 未跟踪 ONNX、checkpoint 和 YOLO 权重 |
| `resources_calibration_ignored.tar.gz` | `resources/calibration/` | 现场标定结果和采集样本 |
| `resources_trajectories_ignored.tar.gz` | `resources/trajectories/` | 未跟踪仿真及真机轨迹、日志 |
| `resources_training_ignored.tar.gz` | `resources/training/` | YOLO 训练输出和 checkpoint |
| `resources_predictions_ignored.tar.gz` | `resources/predictions/` | 离线预测图片 |
| `resources_wheels_ignored.tar.gz` | `resources/wheels/` | 离线安装包 |

`SHA256SUMS` 保存所有压缩包的校验值。压缩包本身已加入 `.gitignore`，应通过移动硬盘、
局域网或其他大文件通道单独传输；恢复脚本和本说明可以随 Git 仓库上传。

这些压缩包没有加密，其中可能包含现场标定和真机轨迹。传输到仓库外部前应确认接收方
和存储位置符合项目的数据管理要求。

## 恢复到仓库

将整个 `offline_artifacts/` 目录复制到目标仓库根目录后执行：

```bash
chmod +x offline_artifacts/restore_ignored_files.sh
./offline_artifacts/restore_ignored_files.sh
```

脚本会先校验 `SHA256SUMS`，再检查所有归档成员都位于 `resources/` 下。默认情况下，只要
目标仓库中已有任意同名文件，就会在解压前整体停止，不会产生部分覆盖。

如果资源包放在仓库外，可显式指定目标仓库：

```bash
/path/to/offline_artifacts/restore_ignored_files.sh /path/to/Ur5e_deploy
```

确认需要用资源包覆盖目标仓库中的同名文件时才使用：

```bash
./offline_artifacts/restore_ignored_files.sh --overwrite /path/to/Ur5e_deploy
```

## 重新打包

在源仓库根目录存在完整 ignored 资源时执行：

```bash
chmod +x offline_artifacts/pack_ignored_files.sh
./offline_artifacts/pack_ignored_files.sh
```

脚本通过 `git ls-files --others --ignored --exclude-standard` 重新收集六类资源，生成新的
`archives/*.tar.gz` 和 `SHA256SUMS`。新生成的压缩包会替换同名旧包；传输前建议再次运行
恢复脚本的 checksum 阶段或直接执行：

```bash
cd offline_artifacts
sha256sum --check SHA256SUMS
```
