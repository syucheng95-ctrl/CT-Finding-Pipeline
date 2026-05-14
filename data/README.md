# 数据目录

放 CT 图像、GT 标注和清单文件。

```
data/
├── images/          ← CT NIfTI（.nii.gz）
├── labels/          ← GT 标注 NIfTI（和 images 对应）
└── *.jsonl          ← 清单文件（每行一个 finding）
```

**清单格式**（JSONL）：
```json
{"id": "finding_001", "case_name": "CT_001.nii.gz", "prompt": "左肺上叶结节", "category": "2d"}
```

- `id`：finding 的唯一 ID
- `case_name`：对应 `data/images/` 下的 CT 文件名
- `prompt`：放射科报告中的病灶描述文本
- `category`：病灶类别代码（如 2d=结节）
