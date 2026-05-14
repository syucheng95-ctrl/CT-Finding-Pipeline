# CT Finding Segmentation Pipeline

完整 CT 病灶分割 pipeline：文本理解 → 解剖裁剪 → 候选生成 → 验证 → 精细分割 → 门控融合。

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 下载模型权重 (~10 GB)

```bash
python download_models.py
```

### 3. 配置

编辑 `config.yaml`，修改数据路径：
```yaml
data:
  ct_images: "你的CT图像目录"
  labels: "你的GT标注目录"
```

### 4. 推理

```bash
python scripts/run_full.py --config config.yaml \
  --stages 0,0.5,1,2_roi,2_eval,gate_table,gate_apply
```

## 流程说明

| Stage | 功能 | 需要 GPU |
|---|---|---|
| 0 | 文本理解 + 解剖裁剪 | 是 |
| 0.5 | 候选区域生成 | 否（CPU） |
| 1 | VoxTell 文本一致性验证 | 是 |
| 2_roi | STU-Net ROI 生成 | 否 |
| 2_eval | STU-Net 精细分割 | 是 |
| gate_table | 生成门控特征表 | 否 |
| gate_apply | 门控决策（S1/S2 选择） | 否 |

## 输出

- `outputs/stage2/finding_preds/` — 最终分割 mask（NIfTI）
- `outputs/final/gate_decisions.csv` — 每个 finding 的 S1/S2 决策
- `outputs/final/final_metrics.json` — 最终 micro Dice 指标

## 训练 Gate（需要 GT 标注）

如果你有 GT 标注的数据集，可以训练自己的 gate：

```bash
# 完整跑 pipeline
python scripts/run_full.py --config config.yaml \
  --stages 0,0.5,1,1_metrics,2_roi,2_eval,gate_table

# 训练 gate
python scripts/run_fit_gate.py

# 评估 gate（5-fold CV）
python scripts/run_cv.py
```

## 可选参数

```bash
--stages 0,0.5,1,2_roi,2_eval     # 只跑特定阶段
--limit-cases 10                    # 限制处理 CT 数量
--manifest ./data/my_manifest.jsonl # 使用自定义清单
```

## 清单格式

JSONL，每行一个 finding：
```json
{"id": "finding_001", "case_name": "CT_001.nii.gz", "prompt": "左肺上叶结节", "category": "2d"}
```

## 故障排除

1. **CUDA out of memory**：减小 `stage2.eval_overlap` 的值
2. **模型下载失败**：检查网络，可按 download_models.py 中的提示手动下载
3. **输出目录权限**：确保 `outputs/` 目录可写
4. **Stage0 太慢**：首次运行 TotalSegmentator 会下载权重 + 处理较慢，后续会快

---

详细模型架构、设计哲学和改进方向见 **[ARCHITECTURE.md](ARCHITECTURE.md)**。
