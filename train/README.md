# 训练脚本参考

这里放了所有模块的训练脚本，供参考学习。**不需要跑，推理 pipeline 不依赖这些文件。**

| 子目录 | 内容 |
|---|---|
| `stage0_router/` | RouterHead 分类器训练 + 数据准备 |
| `stage0_5/` | 候选框生成评估 + 调试脚本 |
| `stage1/` | VoxTell 微调 + baseline 评估 |
| `stage2/` | STU-Net 训练（v1~v4）+ ROI 裁剪 |
| `gate/` | Gate 门控模型训练 + 消融实验 |
