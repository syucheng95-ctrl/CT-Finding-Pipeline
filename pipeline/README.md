# Pipeline 阶段脚本

每个脚本对应 pipeline 的一个阶段。所有脚本都通过 `scripts/run_full.py` 调度，不直接运行。

| 脚本 | 阶段 | 功能 |
|---|---|---|
| `run_stage0.py` | Stage0 | Qwen 文本理解 + TotalSegmentator 解剖裁剪（in-process，模型只加载一次） |
| `run_stage0_5.py` | Stage0.5 | 多专家候选框生成（HU 阈值 / 结节检测 / 兜底） |
| `run_stage1_verify.py` | Stage1 | VoxTell 文本一致性验证，输出 coarse mask |
| `evaluate_stage1_metrics.py` | 1_metrics | 计算 S1 的 micro Dice / Recall / Precision |
| `run_stage2_make_rois.py` | 2_roi | 从 S1 结果生成 ROI 裁剪区域 |
| `run_stage2_eval_finding_level.py` | 2_eval | STU-Net 滑动窗口推理 + overlap/confidence 特征提取 |
| `build_gate_training_table.py` | gate_table | 汇总各阶段特征，生成门控训练表 |
| `_verifier.py` | — | VoxTell 验证器（被 Stage1 调用） |
| `utils.py` | — | 公共工具：config 加载、JSONL 读写 |
