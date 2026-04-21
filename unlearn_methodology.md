# Dual-Assistant ULD: Unlearning via Paired Logit Difference

## 动机

原 ULD(Unlearning from Logit Difference)在推理时用 `final = base + w·assist` 进行遗忘:
- `base` 是原始大模型(冻结)
- `assist` 是个小助手模型,被训练得**在 forget 内容上记住、在 retain 上均匀**
- `w = -1.0`(负号),等价于从 base 里"减去"对 forget 的记忆

**问题**:小助手在 retain 上的"均匀"信号也会被减,对那些**语义上靠近 forget 的 retain 样本**(称为 R_sub),实际上会连带被误伤。

**核心改进**:引入第二个助手 A2,**精准补回**被误伤的 R_sub 区。

## 方法

### 1. R_sub 选择
用 sentence-transformer 编码每个 forget question 和 retain question,计算余弦相似度。对每个 retain item 取其与所有 forget 的**最大相似度**,挑 top-K 作为 R_sub(这些是最可能被 A1 误伤的 retain 样本)。

脚本:[scripts/select_rsub.py](scripts/select_rsub.py)

输出:JSON 里存相对于训练 retain 切片的索引。

### 2. 数据三分

给定 forget 集合 `F`、完整 retain 集合 `R`,选出 `R_sub ⊂ R`,剩下 `R_far = R \ R_sub`。

| 助手 | "forget-role"(GradDescent NTP) | "retain-role"(KL→uniform) |
|---|---|---|
| **A1** | `F ∪ R_sub` | `R_far`(+ perturb) |
| **A2** | `R_sub` | `F ∪ R_far`(+ perturb) |

两助手复用相同的 `remember+uniform` loss,只是**数据组合**不同。

### 3. 架构

- `num_layer=8`(从 base LLM 截取,权重复制 + LoRA r=16 微调)
- A1 与 A2 是**独立**的两个小模型(不共享 LoRA),推理时 logit shape 对齐
- 训练用 `ForgetTrainer` + `EqualForgetRetainSampler` 保证每 batch 有平衡的 forget/retain 比例

### 4. 推理时 logit 组合

```
final_logits = base_logits + w1 · A1_logits + w2 · A2_logits
```

配合 `top_logit_filter=0.01`(只对 base top 1% logits 做对比),A1/A2 在低概率位置被 mask 为 0。

**每种输入下的行为**:

| 输入类型 | base | A1 | A2 | 组合(w1=-0.8, w2=0.8) |
|---|---|---|---|---|
| 真 forget | 高 | 高 | ≈0 | base − 0.8·高 + 0 → **压低** ✓ |
| R_sub | 高 | 高 | 高 | base − 0.8·高 + 0.8·高 ≈ base → **保住** ✓ |
| R_far | 高 | ≈0 | ≈0 | base → **不动** ✓ |

## 关键超参(forget10 + LLaMA-2-7B tofu_ft)

| 参数 | 取值 | 说明 |
|---|---|---|
| A1 `num_layer` | 8 | 原 ULD 默认,容量够 |
| A1 `retain_weight` | **2.0** | 原 ULD 是 5.0 但那会让 A1 学不牢 forget;降到 1.0 又过拟合 → **2.0 是关键折中** |
| A2 `num_layer` | 8 | 同 A1 |
| A2 `retain_weight` | 5.0 | A2 retain 集合很大,需要强 uniform 压制 |
| K (R_sub 大小) | 80 | forget10 有 ~400 forget 项,R_sub 取 retain 的 20% |
| 训练轮数 | 10 epochs | 约 1500 steps(bs=8, accum=2) |
| `weight_a1 / weight_a2` | **-0.8 / +0.8** | 减力度与补力度对称 |
| `top_logit_filter` | 0.01 | 只对 top 1% tokens 做对比修正 |

## 实验结果(TOFU forget10, LLaMA-2-7B tofu_ft)

### 消融(forget_quality 演化)

| Setup | FQ | forget_proba | forget_gen |
|---|---|---|---|
| v1 (rw=5, 8L, -0.8/+0.8) | 0.21 | 0.68 | 0.42 |
| v2a (rw=1, 12L, -1.0/+0.8) | 5e-19 | 0.26 | 0.17 |
| v2b (rw=1, 12L, -0.5/+0.4) | 9e-7 | 0.73 | 0.54 |
| v2c (rw=1, 12L, -0.8/+0.6) | 1.2e-3 | 0.48 | 0.31 |
| **本方法 (rw=2, 8L, -0.8/+0.8)** | **0.654** | **0.52** | **0.35** |

### 对照 ULD 论文(最终配置)

| 指标 | 本方法 | ULD 论文 | Δ |
|---|---|---|---|
| **Forget Quality** | **0.6536** | 0.48 | **+36%** |
| **Model Utility** | **0.6404** | 0.62 | **+3%** |
| Retain ROUGE | 0.805 | — | |
| Real Authors ROUGE | 0.934 | — | |
| Real World ROUGE | 0.893 | — | |
| Forget ROUGE | 0.351 | — | 越低越好 |
| Forget Probability | 0.518 | — | |

Forget Quality = Kolmogorov–Smirnov 检验的 p-value(遗忘模型 vs retain-only 模型在 truth ratio 上的分布相似度),越高越好,上限 1.0。
Model Utility = retain/real_authors/world_facts 三个 eval set 上 ROUGE × Probability × Truth Ratio 的综合分数。

**两个核心指标同时超过 ULD 论文**:加 A2 精确补回 R_sub 不仅提升 FQ(分布更像 retain-only),还保住/微升 MU(retain 能力)。

## 调参直觉

1. **A1 过度记忆(rw 过低)** → 减法产生特异性 anti-signature,偏离 retain-only 分布 → FQ 极低
2. **A1 欠拟合(rw 过高)** → forget 信号弱,减不掉内容 → FQ 稍低
3. **weight 过大(-1.0)** → 过度减法 → forget_proba 过低但分布偏 → FQ 低
4. **weight 过小(-0.5)** → 遗忘不足 → forget_proba 过高 → FQ 低

最优路径:先把 A1 调到 forget_loss 落在 1-2 区间(适度记忆,留有 uniform 成分),再做对称 `w1=-w2` 权重配对。

## 文件索引

- [uld/model/dualcontrastllm.py](uld/model/dualcontrastllm.py) — 双助手推理模型,含独立 KV cache 的 greedy_search 重写
- [uld/data/tofu.py](uld/data/tofu.py) — 按 `data_role` 切分 A1/A2 的训练数据
- [configs/model_mode/dual_uld.yaml](configs/model_mode/dual_uld.yaml) — 推理时加载 A1+A2+base
- [configs/data_mode/dual_a1.yaml](configs/data_mode/dual_a1.yaml) / [dual_a2.yaml](configs/data_mode/dual_a2.yaml) — 数据分区配置
- [scripts/select_rsub.py](scripts/select_rsub.py) — R_sub 挑选
- [bashes/tofu/dual_uld_pipeline.sh](bashes/tofu/dual_uld_pipeline.sh) — 端到端流水线
