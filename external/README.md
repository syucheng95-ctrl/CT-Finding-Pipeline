# 外部适配层

为 pipeline 提供 STU-Net 和 VoxTell 的推理接口。

- `stunet_inference.py`：加载 STU-Net 模型 + 预训练权重，返回滑动窗口推理工具函数
- `voxtell_inference.py`：设置 VoxTell 模块的 import 路径 + 提供 NibabelIO

这两个文件在 pipeline 脚本中通过 `import` 调用，隔离了底层模型加载细节。
