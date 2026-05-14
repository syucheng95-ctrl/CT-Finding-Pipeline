# 入口脚本

`run_full.py` 是全 pipeline 调度器，按顺序运行所有阶段。

**推理模式（默认，不需要 GT）：**
```bash
python scripts/run_full.py --config config.yaml
```

**验证模式（需要 GT，含 gate 重训）：**
```bash
python scripts/run_full.py --config config.yaml --stages 0,0.5,1,1_metrics,2_roi,2_eval,gate_table,gate_fit,gate_apply
```

**只跑后半段：**
```bash
python scripts/run_full.py --stages 2_eval,gate_table,gate_apply
```

- `--limit-cases N` 限制处理 CT 数量
- `--manifest path.jsonl` 指定自定义清单
- `--no-clean` 保留中间 NIfTI 文件
