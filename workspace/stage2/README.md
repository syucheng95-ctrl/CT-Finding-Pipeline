# Stage2：STU-Net 精细分割

在 Stage1 确认的 ROI 区域上，用 STU-Net 进行精细的 3D 分割。

- `model.py`：`create_stunet_model()`，基于 MedIM 骨架的 STU-Net-S
- `eval_stratified.py`：滑动窗口推理工具函数（`pad_to_shape_centered`、`prepare_image`、`load_group_patch_shapes`）
- `dataset.py`：`get_group()` 根据形状分组 + `GROUP_ORDER`

权重加载不依赖 HuggingFace 网络连接（`pretrained_dataset=None`），完整权重包含在 `models/stunet/best.pt` 中。
