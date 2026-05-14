# VoxTell 核心代码

VoxTell 是一个文本引导的 3D 医学图像分割模型。

- `voxtell/model/voxtell_model.py`：VoxTellModel 主模型
- `voxtell/model/transformer.py`：TransformerDecoder 和 TransformerDecoderLayer
- `voxtell/inference/predictor.py`：VoxTellPredictor（滑动窗口推理、预处理、文本 embedding）
- `voxtell/utils/text_embedding.py`：文本池化 `last_token_pool` 和指令包装函数

模型权重在 `models/voxtell_v1.1/` 下，由 `workspace/stage1/` 的代码加载。
