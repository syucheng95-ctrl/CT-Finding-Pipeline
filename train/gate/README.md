# Gate 门控训练

- `train_gate.py`：主力训练脚本，支持 5-fold CV 和 `--fit-final` 全量训练
- `train_gate_v3.py`：回归目标对比实验（分类 vs 回归 vs 加权）
- `train_gate_ablation.py` / `train_gate_ablation_v2.py`：特征消融实验
- `train_gate_kmeans.py`：KMeans 聚类特征实验
- `train_gate_search.py`：多配置 + 多种子稳定性搜索
- `analyze_gate_errors.py`：FP/FN 逐 finding 错误分析
- `build_gate_training_table.py`：生成门控训练特征表
- `apply_gate.py`：推理脚本（也在 `gate/` 目录下用于生产）
