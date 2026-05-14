# Stage1：VoxTell 文本一致性验证

加载 VoxTell 模型，对每个 proposal 区域进行文本引导的分割，验证 proposal 内是否真的有病灶。

- `config.py`：构建 VoxTell 模型（`build_voxtell_from_checkpoint`），加载原始 checkpoint
- `eval.py`：`DoRAEvalPredictor`（推理包装器，滑动窗口 + 文本 embedding）

被 `pipeline/run_stage1_verify.py` 和 `_verifier.py` 调用。
