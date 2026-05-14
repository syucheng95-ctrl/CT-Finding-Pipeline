# Stage0：文本理解 + 解剖裁剪

对每个 CT 的每个 finding，用 Qwen3-Embedding-4B 提取文本特征，RouterHead 分类器预测类别和松紧度，TotalSegmentator 输出肺叶分割，最后按策略裁出 ROI 区域。

**入口**：`run_stage0_policy_crop.py` → `process_one_case()`（in-process 调用，模型只加载一次）
**关键依赖**：`build_qwen_embeddings.py`（Qwen embedding）、`anatomy_expert.py`（TotalSegmentator）、`predict_router.py`（RouterHead）、`router_policy.py`（fail-open 逻辑）
