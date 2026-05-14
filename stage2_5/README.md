# Stage2.5：ROI 量化分析 + 标准化报告

独立的 ROI 后处理模块，**不接入 run_full 默认链路**。

**功能**：
- 从 Stage2 输出的 ROI mask 提取量化指标（肺透光度、气道变化、胸膜征、病灶体积等）
- 用 Cross-Attention 分类器对 ROI 做进一步的 14 类分类
- 生成标准化临床报告

**用法**：
```bash
python -m stage2_5.pipeline
```

**依赖**：需要 `outputs/stage2/roi_manifest.jsonl` 和 `outputs/stage2/roi_images/`（由 Stage2 生成）。分类器权重需要单独训练或下载。

- `classifier/`：Cross-Attention 模型（训练 + 推理）
- `metrics/`：量化指标计算（肺实质、气道、胸膜等）
- `pipeline.py`：主流程入口
- `report.py`：报告生成
