# Isaac Gym 成功权重集合

该集合包含当前 `train_dir` 中，在统一 Isaac Gym 抛投评估里至少出现一次成功事件（`max_successes > 0`）的唯一权重。

- 评估条件：`seed=0`、16 个环境、确定性推理、统一的 `AllegroRokaeLSTM` 抛投任务配置。
- 去重方式：按完整 SHA-256 去重。
- 排序方式：先按 `max_successes` 降序，再按平均 reward 降序。
- 文件名：`rank_来源_训练目标值_maxsucc_平均reward_SHA前缀.pth`。
- 原始权重均保留，集合目录中的文件是副本。
- `max_successes` 是此次有限环境评估中观测到的最大成功计数，并非带置信区间的统计成功率。

完整来源、哈希及评估指标见 `successful_weights_manifest.csv`。
