# Stage0 Router 训练

- `train_router_head.py`：训练 RouterHead 分类器（预测 finding 类别和裁剪松紧度）
- `prepare_router_data.py`：生成训练数据（Qwen embedding + 标签）
- `router_model.py`：RouterHead 模型定义
- `evaluate_router.py`：评估路由准确率
- `predict_router.py`：推理入口
