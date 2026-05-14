# Stage0.5：多专家候选框生成

在 Stage0 裁剪后的 ROI 上，用多个专家模块生成候选病灶区域。

- `hu_expert.py`：基于 HU 阈值对比度
- `nodule_detector_expert.py`：MONAI 肺结节检测模型
- `diffuse_expert.py`：兜底专家（弥散病灶）
- `category_router.py`：根据类别选择专家
- `proposal_fusion.py`：候选框融合去重
- `morphology_filter.py` + `connected_components.py`：形态学后处理

**入口**：`run_stage0_5.py`，作为 `proposal_generator` 包通过 `python -m` 调用
