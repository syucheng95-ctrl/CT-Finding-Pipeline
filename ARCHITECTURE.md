# Pipeline 架构

## 模型架构框图

```
                          CT Image (.nii.gz)         ← CT 图像
                                 │
                    ┌────────────┴────────────┐
                    │     Stage0 — Routing    │   文本理解 + 解剖裁剪
                    │  ┌───────────────────┐  │
                    │  │  Qwen3-Embedding  │  │   文本 embedding（7.7 GB）
                    │  │        ↓          │  │
                    │  │  RouterHead       │  │   14 类 + 松紧度预测
                    │  │        ↓          │  │
                    │  │  TotalSegmentator │  │   肺叶分割 + 解剖裁剪
                    │  └───────────────────┘  │
                    └────────────┬────────────┘
                                 │
                    Cropped ROI CT + Text      ← 裁剪后的 ROI + 文本
                                 │
                    ┌────────────┴────────────┐
                    │ Stage0.5 — Proposals    │   多专家候选框生成
                    │  ┌───────────────────┐  │
                    │  │  HU Expert        │  │   CT 值对比度
                    │  │  MONAI Nodule     │  │   深度学习结节检测
                    │  │  Diffuse Fallback │  │   弥散兜底
                    │  └───────────────────┘  │
                    └────────────┬────────────┘
                                 │
                      Proposal BBox List       ← proposal bbox 列表
                                 │
                    ┌────────────┴────────────┐
                    │  Stage1 — Verification  │   文本一致性验证
                    │  ┌───────────────────┐  │
                    │  │  VoxTell          │  │   文本引导分割（1.7 GB）
                    │  │  DoRA Predictor   │  │   滑动窗口推理
                    │  │        ↓          │  │
                    │  │  Coarse Mask      │  │   粗分割 + 概率图
                    │  └───────────────────┘  │
                    └────────────┬────────────┘
                                 │
                  Verified ROI + Coarse Mask   ← verified ROI + coarse mask
                                 │
                    ┌────────────┴────────────┐
                    │ Stage2 — Fine Segment   │   精细分割
                    │  ┌───────────────────┐  │
                    │  │  STU-Net-S        │  │   3D 滑动窗口（167 MB）
                    │  │  MedIM Backbone   │  │   预训练骨架
                    │  │        ↓          │  │
                    │  │  Raw S2 Mask      │  │   精细分割 + logits
                    │  │  Overlap/Conf Feat│  │   S1∩S2 + 置信度特征
                    │  └───────────────────┘  │
                    └────────────┬────────────┘
                                 │
              S1 Mask + S2 Mask + Features     ← S1 + S2_raw + 特征
                                 │
                    ┌────────────┴────────────┐
                    │     Gate — Decision     │   门控决策
                    │  ┌───────────────────┐  │
                    │  │  RandomForest     │  │   93 特征二分类
                    │  │  Per Finding:     │  │   对每个 finding：
                    │  │  Pick S2 or S1    │  │   use_s2 ? S2 : S1
                    │  └───────────────────┘  │
                    └────────────┬────────────┘
                                 │
                     Final Segmentation       ← 最终分割 mask
                                 │
                    ┌────────────┴────────────┐
                    │ Stage2.5 — Quant        │   量化分析（独立模块）
                    │  ┌───────────────────┐  │
                    │  │  12 ROI Metrics   │  │   体积/直径/HU/毛刺...
                    │  │  Router Category  │  │   14 类 → metric group
                    │  │  Standard Report  │  │   JSON 输出
                    │  └───────────────────┘  │
                    └─────────────────────────┘
```

## 设计哲学

**核心问题**：VoxTell 是一个文本引导的分割模型，但它没有理解全 CT 全局上下文的能力——直接把 512×512×200 的全 CT 送进去，它不知道病灶在哪，只能盲目滑动窗口，效果差（FullCT VoxTell Dice 仅 0.2244）。

**解法**：先用 Stage0（文本理解 + 解剖裁剪）+ Stage0.5（多专家候选框）把 CT 缩小到病灶附近的小区域，再让 VoxTell 在小区域内做文本引导推理。本质是**先预定位，再精细分割**。

Pipeline 从 FullCT VoxTell 的 0.2244 提升到 S1 hybrid 的 0.2334，证明了这个分层设计的有效性。

---

## 各阶段详细说明

### Stage0 — 文本理解 + 解剖裁剪

**设计逻辑**：Qwen 读文本 → RouterHead 判断病灶在哪、什么类型 → TotalSegmentator 识别肺叶 → 按策略裁出焦点区域。本质是"文本引导的 ROI 初筛"，为下游 VoxTell 提供局部上下文。

| 模型 | 大小 | 输入 | 输出 |
|---|---|---|---|
| Qwen3-Embedding-4B | 7.7 GB | finding 文本（放射报告） | 2560 维文本向量 |
| RouterHead | 9 MB | 文本向量 | 14 类 + 松紧度 |
| TotalSegmentator | ~300 MB | 全 CT | 肺叶分割 mask |

**流程**：对每个 finding 的文本 prompt 进行 Qwen embedding → RouterHead 分类器预测类别和裁剪松紧度 → TotalSegmentator 做肺叶分割 → 按策略裁剪 CT。

**优化**：in-process 调用，模型只加载一次，100 case 从 ~3h 降到 ~1.5h。

### Stage0.5 — 多专家候选框生成

**设计逻辑**：不同病灶视觉特征差异大——肺气肿靠 HU 阈值，结节靠 MONAI 检测器，弥散病变兜底走全裁。Router 类别决定走哪个专家。MoE 模式，比单一检测方法覆盖面广。

| 专家模块 | 方式 |
|---|---|
| HU 阈值专家 | CT 值对比度分析 |
| MONAI 肺结节检测器 | 深度学习模型（160 MB） |
| 弥散兜底专家 | 全 ROI 裁剪 |

### Stage1 — VoxTell 文本验证

**设计逻辑**：Stage0.5 候选框多但假阳性也高。VoxTell 是文本引导分割模型——它读图像 + 文本，只分割文本描述的那种病灶。如果 candidate bbox 里没有文本说的东西，VoxTell 基本不分或者概率很低。本质是"用文本过滤假阳性"，帮昂贵的 Stage2 省力。

| 模型 | 大小 | 功能 |
|---|---|---|
| VoxTell | 1.7 GB | 文本引导的 3D 分割 |

**Hybrid 路由**：
- 1a/1b/1c/2f → FullCT VoxTell（全 CT 推理）
- 1e/2a/2b/2d/2g → VoxTell mask（ROI 推理）
- 其他 → Stage0.5 bbox

### Stage2 — STU-Net 精细分割

**设计逻辑**：VoxTell coarse mask 分辨率低，边界粗。STU-Net 在确认的 ROI 上做精细体素级分割。同时提取 overlap 和 confidence 特征——这是 Gate 最需要的信号（S2 跟 S1 差多远，S2 自己有多确定）。

| 模型 | 大小 | 功能 |
|---|---|---|
| STU-Net-S + MedIM | 167 + 112 MB | 3D 滑动窗口分割 |

### Gate — 门控决策

**设计逻辑**：S1 和 S2 各有所长——S2 弥漫病变容易 over-segment，局灶病变好；S1 稳定但不精细。Gate 用 93 个推理可得的特征学"什么时候信 S2"，不是简单比 Dice。

| 模型 | 大小 | 输入 | 输出 |
|---|---|---|---|
| RandomForest | 52 KB | 93 维特征向量 | use_s1 / use_s2 |

**训练**：5-fold GroupKFold，Dinkelbach 迭代构造 micro-optimal label。

### Stage2.5 — 量化分析（独立模块）

**设计逻辑**：分割 mask 本身不是终点，医生需要量化——体积、直径、HU、毛刺指数等。从最终 mask 提取 12 项指标，Router 类别路由到对应指标组，输出标准化报告。

---

## 特征体系（Gate 输入）

| 特征组 | 数量 | 来源 |
|---|---|---|
| 文本关键词 | 21 | finding prompt 规则匹配 |
| Stage0 Router | 13 | 类别、解剖、策略、置信度 |
| Crop 几何 | 16 | 裁剪 bbox 位置/贴边/体积 |
| Stage0.5 Proposal | 15 | 候选框数量/体积/质量 |
| Stage1 Verifier | 15 | 验证通过/拒绝、概率/前景/分数 |
| S1 Coarse | 5 | S1 预测体积/对数/空/大 |
| Stage2 ROI | 12 | ROI 数量/体积/来源 |
| S2 Raw Prediction | 8 | S2 体积/比例/标记 |

## 数据流

```
data/images/         ← CT 图像
data/labels/         ← GT 标注（训练 gate 需要）
data/*.jsonl         ← 清单文件
        │
        ├── Stage0   → outputs/stage0/
        ├── Stage0.5 → outputs/stage0_5/
        ├── Stage1   → outputs/stage1/
        ├── Stage2   → outputs/stage2/
        ├── Gate     → outputs/final/
        └── Stage2.5 → stage2_5/outputs/
```

## 性能对比

| 方案 | Dice | Recall | Precision |
|---|---|---|---|
| FullCT VoxTell（th=0.5） | 0.2244 | 0.2455 | 0.2066 |
| S1 hybrid pipeline | 0.2334 | 0.3022 | 0.1901 |
| Learned gate | **0.2417** | 0.3009 | 0.2020 |

**其他指标**：

| 指标 | 值 |
|---|---|
| Gate micro Dice（5-fold CV） | 0.2373 |
| Gate vs S1 提升 | +0.008 |
| Gate 选 S2 比例 | ~37/278（13%） |
| 1 case 推理时间 | ~4 分钟（GPU RTX 4080） |

---

## 改进方向

### Stage0.5：候选框生成可更精细

当前 HU Expert 仅基于 CT 值对比度做简单的阈值分割，对复杂病灶（如 GGO、混合密度结节）定位不够精准。可考虑：
- 引入更多专家模块（如专门处理 GGO 的密度梯度专家）
- 类别路由从粗粒度 14 类 → 更细的子类
- RU expert 内部做自适应阈值，而非固定 HU 窗口

### Stage1：VoxTell 微调（已尝试，效果不佳）

曾在有 GT 的数据集上对 VoxTell 做 DoRA 微调，但下游 Dice 无明显提升。原因可能是：VoxTell 原始 checkpoint 已在足够大的数据集上训练，当前 278 finding 数据量不足以带来增益。后续数据量增大后可重新尝试。

### Stage2：STU-Net 需要域内微调

当前 STU-Net 在 TotalSegmentator 预训练权重上直接推理，未在目标 CT 数据上微调。后续应在 pipeline 输出的真实 ROI 上做 fine-tuning，让模型适应 pipeline 的输入分布（而非仅依赖公开预训练特征）。

### Gate：数据量瓶颈

当前 Gate 仅 278 findings（~100 CT cases），正类（应选 S2）约 42 个。RF 虽已超越 S1，但提升幅度小（+0.008）、跨 fold 方差大。后续数据量上来后可尝试：
- 更大特征集（overlap/confidence 在更多样本下可能有用）
- XGBoost / LightGBM
- 简单神经网络（100+ 样本）

### Stage1：Hybrid 路由阈值可 per-category 优化

当前 FullCT VoxTell 和二值 coarse mask 的概率阈值统一为 0.3（即 prob > 0.3 判定为前景）。但不同类别的最优阈值差异很大：实性结节（2d）可能 0.2 就够了，磨玻璃（2c）可能需 0.5 才能减少噪声。目前的统一 0.3 是在早期验证集上手动选取的，未做系统性调优。

后续可对每类单独扫阈值，选 per-category 最优 Dice，存储到配置中，推理时按 Router 预测的类别动态选取。

### Pipeline 容错与断点续跑

当前 `run_full.py` 任一 stage 报错直接退出（`sys.exit`）。实际部署中会遇到 CT 读取失败、OOM、单个 finding 推理超时等异常场景。需要：

- 单个 finding 失败 → skip + 记录到日志，继续处理下一个
- 整个 stage 异常 → 保存已完成 stage 的状态，支持从断点重跑（`--resume-from stage1`）
- GPU OOM → 自动降低 overlap 参数重试

### 外部验证泛化能力

当前所有指标（S1=0.2334, Gate=0.2417）均在自有的 val_20 验证集上测得，278 findings 约 100 case。Gate 的训练和评估共享同一批数据的特征分布，CV 结果（0.2373）可能仍偏乐观（case 级分组可缓解但无法消除数据分布偏差）。

后续需要：
- 完全独立的 hold-out 测试集（非同一批采集的 CT）
- 跨中心数据测试（不同医院、不同机型）
- 在外部数据上重新评估 Gate 的泛化能力，确认 CV 估计的 0.2373 是否在真实场景中成立

### Stage2.5：指标校准与参考范围

当前 12 项量化指标（体积、直径、HU、毛刺指数等）输出了原始数值，但未提供"正常/异常"判断标准。实际医生使用时需要知道某个值是否在正常范围内。

后续可：
- 对不同病灶类型设定 per-category 参考范围（如 2d 结节 < 8mm 为微小结节，> 30mm 为肿块）
- 毛刺指数、空洞比等形态学指标需要有放射科专家标注的 ground truth 做校准
- 报告里加入"与上一期对比"功能（需要随访数据）
