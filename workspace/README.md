# 推理源码

包含 5 个推理模块，均为原始 workspace 源码适配版本（路径全部改为自包含）。

| 模块 | 入口 | 功能 |
|---|---|---|
| `stage0/` | `run_stage0_policy_crop.py` | Qwen 文本 embedding + Router 分类 + TotalSegmentator 解剖裁剪 |
| `proposal_generator/` | `run_stage0_5.py` | 多专家候选框生成（HU、MONAI 结节、兜底） |
| `stage1/` | `config.py` + `eval.py` | VoxTell 模型构建 + DoRA 推理 |
| `stage2/` | `model.py` + `eval_stratified.py` | STU-Net 模型定义 + 滑动窗口推理 |
| `voxtell/` | `voxtell/` | VoxTell 核心模型 + 预测器 + 文本 embedding |

所有模块互相独立，不依赖任何外部目录。
