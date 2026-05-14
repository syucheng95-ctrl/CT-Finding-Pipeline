# Stage1 VoxTell 训练

- `train.py`：VoxTell 微调脚本（最终未采用，推理直接使用原始 VoxTell checkpoint）
- `eval_base_voxtell.py`：FullCT VoxTell baseline 评估（全 CT 滑动窗口 + 文本引导分割）
- `precompute_embeddings.py`：预计算 Qwen 文本 embedding 加速训练
