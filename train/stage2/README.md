# Stage2 STU-Net 训练

- `train.py` / `train_v2.py` / `train_v3.py` / `train_v4.py`：STU-Net 训练脚本的四个版本迭代
- `build_manifest.py`：构建训练清单
- `recrop_v3_rois.py` / `recrop_v3_rois_stream.py`：ROI 裁剪（固定尺寸 + 比例）
- `run_recrop_v3_full.py`：批量 ROI 裁剪
