# Gate 门控模块

判断每个 finding 用 S1（粗分割）还是 S2_raw（精细分割）。

- `train_gate.py`：训练 RandomForest 门控模型，支持 `--fit-final`（全量训练并保存）和默认的 5-fold CV
- `apply_gate.py`：加载训练好的模型进行推理，输出每个 finding 的 S1/S2 决策
- `gate_config.json`：门控配置（特征筛选、阈值、drop_prefixes）
- `outputs/`：保存 gate_model.joblib（模型权重）、gate_training_table.csv（特征表）
