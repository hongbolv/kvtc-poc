# kvtc-poc 项目深度研究报告

## 1. 项目概述

本项目是 NVIDIA **kvtc**（KV Cache Transform Coding）论文 ([arXiv:2511.01815](https://arxiv.org/abs/2511.01815), ICLR 2026) 的概念验证 (PoC) 实现，针对 **Llama 3.2 1B** 模型进行 KV 缓存压缩。

### 1.1 解决的核心问题

在大语言模型 (LLM) 的自回归推理过程中，模型会缓存每层注意力的 Key 和 Value 投影（即 **KV 缓存**），避免对每个新 token 重复计算。随着上下文长度增长，KV 缓存会大量占用 GPU 显存 —— 这些显存本可以服务其他请求。

kvtc 使用源自图像/视频压缩的技术（类似于"LLM 记忆的 JPEG"），在保持输出质量几乎不变的情况下，实现 **~5-20× 的大小缩减**。

### 1.2 定位与限制

kvtc **不是** vLLM FP8 KV 缓存量化的替代品（FP8 在推理过程中零成本提供 2× 压缩）。kvtc 专为 **存储和卸载** 场景设计 —— 在对话轮次之间、跨节点传输时、或将缓存卸载至 CPU/SSD 时压缩 KV 缓存以释放 GPU 显存。

### 1.3 仓库文件结构

| 文件 | 描述 |
|---|---|
| `kvtc_poc.py` | 核心实现 — 校准、压缩、重建质量报告（CPU zlib DEFLATE） |
| `kvtc_poc_gpu.py` | 与上述相同，但增加了 **nvCOMP GPU DEFLATE** 支持 |
| `kvtc_rag_poc.py` | RAG 多轮对话模拟，对比 3 种策略：重计算 vs 持有在 HBM vs kvtc 压缩/解压缩 |

---

## 2. 核心工作原理

### 2.1 压缩管线总览

```
压缩流程:
  KV Cache → 移除 RoPE → PCA 投影 → 量化 → DEFLATE → 压缩存储

解压缩流程:
  压缩存储 → Inflate → 反量化 → 逆 PCA 投影 → 重新应用 RoPE → KV Cache
```

整个系统分为三个阶段：
1. **校准 (Calibration)** — 对模型运行一次，收集 KV 缓存样本，学习 PCA 基
2. **压缩 (Compress)** — 将数据投影到紧凑空间，使用可变精度量化，再进行熵编码
3. **解压缩 (Decompress)** — 反向执行管线，恢复 KV 缓存

### 2.2 目标模型架构参数

本 PoC 针对 **Llama 3.2 1B** 模型，关键参数如下：

```python
MODEL_ID = "meta-llama/Llama-3.2-1B"
N_LAYERS = 16          # Transformer 层数
N_KV_HEADS = 8         # KV 注意力头数
HEAD_DIM = 64          # 每头维度
CROSS_LAYER_DIM = 16 * 8 * 64 = 8192  # 跨层拼接后的维度
```

### 2.3 RoPE（旋转位置编码）处理

**为什么要移除 RoPE？**
RoPE（Rotary Position Embedding）会对 Key 向量施加位置相关的旋转变换，这种变换会**破坏 PCA 可利用的低秩结构**。因此在压缩前必须移除 RoPE，解压缩后再重新应用。

**Llama 3 的分段缩放 RoPE 实现：**

```python
def compute_llama3_inv_freq(head_dim, device):
    # 基础逆频率
    base_inv = 1 / (θ^(2i/d))  # θ = 500000.0
    # 缩放逆频率
    scaled_inv = base_inv / ROPE_FACTOR  # ROPE_FACTOR = 32.0
    # 根据波长区间选择：
    #   短波长 (< high_wl) → 使用原始 base_inv
    #   长波长 (> low_wl)  → 使用缩放后的 scaled_inv
    #   中间区间           → 平滑插值
```

**向量化 RoPE 移除/应用：**

```python
def remove_rope_batch(keys, cos, sin):
    # 逆 RoPE: keys * cos - rotate_half(keys) * sin
    # keys: [n_pos, n_heads, head_dim]，全部位置同时处理

def apply_rope_batch(keys, cos, sin):
    # 正向 RoPE: keys * cos + rotate_half(keys) * sin
```

`rotate_half(x)` 将向量后半部分取负移至前半部分，实现 RoPE 所需的旋转操作。

### 2.4 跨层拼接 (Cross-Layer Concatenation)

KV 缓存在不同注意力层和头之间存在**强相关性**。kvtc 的核心洞察是将所有层的 KV 向量跨层拼接后再进行 PCA，以充分利用这种相关性：

```
对于每个 token 位置 t:
  keys_mat[t] = concat(layer_0_heads, layer_1_heads, ..., layer_15_heads)
             = [n_pos, 16层 × 8头 × 64维] = [n_pos, 8192]
```

Key 和 Value 分别进行跨层拼接，各自独立压缩。

### 2.5 PCA 投影

**主成分分析 (PCA)** 将高维数据投影到最优的低维子空间：

```python
def pca(data, rank):
    mu = data.mean(dim=0)              # 均值中心化
    centered = data - mu
    U, S, V = torch.pca_lowrank(centered, q=rank, niter=5)
    return V, mu, S  # V: 投影矩阵 [8192, rank], mu: 均值, S: 奇异值
```

- `V` (投影矩阵): 将 8192 维数据投影到 `rank` 维空间（默认 rank ≤ 4096）
- `mu` (均值向量): 用于中心化
- `S` (奇异值): 表示每个主成分的重要性，用于后续的比特分配

### 2.6 DP 比特分配 (Dynamic Programming Bit Allocation)

这是 kvtc 的关键创新之一。DP 算法决定每组 PCA 分量应分配多少比特来量化，目标是在固定比特预算下**最小化 Frobenius 重建误差**。

**算法参数：**
- 分组大小 `GROUP_SIZES = [16, 64, 256, 1024]`
- 每元素比特数 `BIT_OPTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8]`
- 每个激活组的固定开销 `OVERHEAD = 32 bits`（用于存储缩放参数）
- 0 比特 = 完全丢弃该组（无开销）

**比特预算计算：**
```python
budget = total_features * 16 / target_cr
# 例: 8192 * 16 / 16 = 8192 bits per token position
```

**贪心 DP 策略：**
1. 对每种分组大小，计算每组在每种比特等级下的 "效率" = 方差增益 / 比特成本
2. 按效率降序排列，贪心地分配比特预算
3. 选择总误差最小的分组大小

**量化误差模型：**
```python
def quant_error(group_var, bits):
    if bits == 0: return group_var        # 全部丢弃
    return group_var / (4.0 ** bits)       # 均匀量化误差 ≈ var / 4^b
```

**典型结果：** 领先的主成分获得 8 比特（最高精度），尾部成分获得 0 比特（丢弃）。这符合论文 Figure 6 的描述。

### 2.7 量化与反量化

使用简单的**均匀量化**：

```python
def _quant(vals, n_bits):
    n_levels = (1 << n_bits) - 1     # 例: 8 bits → 255 个量化级别
    vmin, vmax = vals.min(), vals.max()
    scale = (vmax - vmin) / n_levels
    q = round((vals - vmin) / scale)  # 量化为整数
    return q, vmin, scale

def _dequant(q, vmin, scale):
    return q * scale + vmin           # 恢复浮点值
```

每个激活的 PCA 分量存储 11 字节头部：
- `idx` (2B): 分量索引
- `bits` (1B): 比特数
- `vmin` (4B): 最小值
- `scale` (4B): 缩放因子

### 2.8 熵编码

量化后的数据通过 **DEFLATE** 算法进行无损压缩，进一步减少存储空间：

| 版本 | 后端 | 说明 |
|---|---|---|
| `kvtc_poc.py` | `zlib` (CPU) | Python 标准库，纯 CPU |
| `kvtc_poc_gpu.py` | `nvCOMP` (GPU) / `zlib` 回退 | NVIDIA GPU 加速 DEFLATE |
| `kvtc_rag_poc.py` | `nvCOMP` / `zlib` 自动选择 | 同上 |

### 2.9 Sink Token 和 Window 处理

KV 缓存并非全部压缩。序列被分为三个区域：

```
[sink_tokens | 压缩区域 | window_tokens]
     4个          中间部分      128个

- sink_tokens (默认 4): 前几个 token 获得不成比例的注意力权重，
                        保持未压缩以保证精度
- window_tokens (默认 128): 最近的 token 保持未压缩，类似滑动窗口
- 压缩区域: 中间的 token 通过 kvtc 管线压缩
```

---

## 3. 完整调用流程

### 3.1 校准流程 (`KVTCCalibrator.calibrate`)

```
main()
 ├── 加载模型和分词器
 ├── 准备校准文本 (128个样本, 混合短长文本)
 └── KVTCCalibrator.calibrate(model, tokenizer, texts)
      ├── collect(model, tokenizer, texts, max_len=2048)
      │    └── 对每个校准文本:
      │         ├── 分词并前向推理获取 KV 缓存
      │         ├── extract_kv(past_key_values) → pk[16层], pv[16层]
      │         ├── build_rope_cache(seq_len, head_dim, device)
      │         ├── 提取非 sink 区域: pk[li][0,:,s:,:] → [n_pos, 8, 64]
      │         ├── remove_rope_batch(keys, cos, sin)  # 向量化移除 RoPE
      │         └── 跨层拼接: [n_pos, 16*8*64] = [n_pos, 8192]
      │         → 返回 all_keys [N, 8192], all_values [N, 8192]
      │
      ├── pca(keys_data, rank=4096)
      │    ├── 均值中心化
      │    └── torch.pca_lowrank → V_k[8192, rank], mu_k[8192], sigma_k[rank]
      │
      ├── pca(values_data, rank=4096)
      │    └── → V_v, mu_v, sigma_v
      │
      ├── dp_alloc(sigma_k, 8192, "Keys")
      │    └── 贪心 DP → alloc_k[rank] (每个分量的比特数)
      │
      └── dp_alloc(sigma_v, 8192, "Values")
           └── → alloc_v[rank]
```

校准产出物（per-model，可复用）：
- `V_k`, `V_v`: PCA 投影矩阵
- `mu_k`, `mu_v`: 均值向量
- `alloc_k`, `alloc_v`: 比特分配方案

### 3.2 压缩流程 (`KVTCCompressor.compress_full`)

```
compress_full(past_key_values)
 ├── extract_kv(past) → pk[16层][1,8,sl,64], pv[16层][1,8,sl,64]
 ├── 确定区域边界: sink=4, window=128, cs=4, ce=sl-128
 ├── build_rope_cache(sl, head_dim, device)
 │
 ├── 提取压缩区域 (对每层):
 │    ├── pk[li][0,:,cs:ce,:].permute → [nc, 8, 64]
 │    └── pv[li][0,:,cs:ce,:].permute → [nc, 8, 64]
 │
 ├── 向量化 RoPE 移除:
 │    └── remove_rope_batch(key_layers[li], cos[cs:ce], sin[cs:ce])
 │
 ├── 跨层拼接:
 │    ├── keys_mat = cat(所有层 keys) → [nc, 8192]
 │    └── vals_mat = cat(所有层 values) → [nc, 8192]
 │
 ├── compress_matrix(keys_mat, V_k, mu_k, alloc_k)
 │    ├── PCA 投影: D = (data - mu) @ V → [nc, rank]
 │    ├── 对每个激活分量 (alloc[idx] > 0):
 │    │    ├── _quant(D[:, idx], bits) → q, vmin, scale
 │    │    └── pack header (11B) + 量化数据
 │    └── zlib.compress(raw_bytes) → compressed_bytes
 │
 ├── compress_matrix(vals_mat, V_v, mu_v, alloc_v) → 同上
 │
 ├── 保存未压缩区域:
 │    ├── sink: pk[li][:,:,:4,:].clone()
 │    └── window: pk[li][:,:,ce:,:].clone()
 │
 └── 返回 {type:"kvtc", ck, cv, sink, win, cs, ce, sl, metrics}
```

### 3.3 解压缩流程 (`KVTCCompressor.decompress_full`)

```
decompress_full(compressed_result)
 ├── build_rope_cache(sl, head_dim, device)
 │
 ├── decompress_matrix(ck, V_k, mu_k, alloc_k)
 │    ├── zlib.decompress(compressed_bytes)
 │    ├── 对每个激活分量:
 │    │    ├── unpack header → idx, bits, vmin, scale
 │    │    ├── 读取量化数据 → numpy array
 │    │    └── _dequant(q, vmin, scale) → D[:, idx]
 │    └── 逆投影: data = D @ V.T + mu → [nc, 8192]
 │
 ├── decompress_matrix(cv, V_v, mu_v, alloc_v) → 同上
 │
 ├── 对每层 li = 0..15:
 │    ├── 从 8192 维向量中切片该层的数据:
 │    │    fs = li * 8 * 64, fe = fs + 512
 │    │    lk = km[:, fs:fe].reshape(nc, 8, 64)
 │    │    lv = vm[:, fs:fe].reshape(nc, 8, 64)
 │    ├── apply_rope_batch(lk, cos[cs:ce], sin[cs:ce])  # 重新应用 RoPE
 │    ├── 维度转换: [nc, 8, 64] → [1, 8, nc, 64]
 │    └── 拼接: cat(sink, compressed, window) → [1, 8, sl, 64]
 │
 └── 返回 rk_list[16层], rv_list[16层]
```

### 3.4 重建质量评估 (`recon_error`)

```
recon_error(original_past, reconstructed)
 ├── 对每层计算:
 │    ├── MSE (均方误差): F.mse_loss(recon, orig)
 │    ├── 余弦相似度: F.cosine_similarity(orig, recon, dim=-1)
 │    └── 相对误差: ||recon - orig|| / ||orig||
 └── 返回 16 层的平均值
```

---

## 4. 三个文件的差异分析

### 4.1 `kvtc_poc.py` — CPU 基线版本

- 仅使用 `zlib` CPU DEFLATE
- 完整的代码注释和结构化输出
- 适合理解算法和在无 GPU 环境运行
- 方法命名: `compress_full()`, `decompress_full()`

### 4.2 `kvtc_poc_gpu.py` — GPU 加速版本

- 新增 `nvidia.nvcomp` GPU DEFLATE 支持
- 自动回退到 `zlib` 如果 nvCOMP 不可用
- 在 `compress_matrix` 和 `decompress_matrix` 中增加了 GPU 路径：
  ```python
  if HAS_NVCOMP:
      raw_tensor = torch.frombuffer(raw, dtype=torch.uint8).cuda()
      nv_arr = nvcomp.as_array(raw_tensor)
      codec = nvcomp.Codec(algorithm="Deflate")
      comp_arr = codec.encode(nv_arr)
  ```
- 其余算法逻辑与 CPU 版完全一致

### 4.3 `kvtc_rag_poc.py` — RAG 多轮对话模拟

这是最接近生产场景的示范，模拟了 RAG 工作流：

**三种策略对比：**

| 策略 | 描述 | TTFT | 显存占用 |
|---|---|---|---|
| **A: Recompute** | 每轮丢弃缓存，重新计算完整上下文 | 最慢 (完整 prefill) | 0 (无缓存保留) |
| **B: Hold** | 将 KV 缓存保留在 GPU HBM 中 | 最快 (仅新 token prefill) | 最高 (缓存常驻) |
| **C: kvtc** | 压缩缓存存储，下轮解压恢复 | 中等 (解压 + 新 token) | 低 (仅存储压缩数据) |

**关键差异：**
- 方法命名简化: `compress()`, `decompress()` (不带 `_full` 后缀)
- 使用 `DynamicCache` 恢复解压缩的缓存用于后续推理
- 包含 GPU 显存监控 (`torch.cuda.memory_allocated()`)
- 使用 `model.generate()` 进行实际生成
- 内置 5 个 RAG 文档 + 5 个问答对用于多轮模拟

---

## 5. 数据流图

### 5.1 校准阶段数据流

```
128 个校准文本
    │
    ▼
┌─────────────────────────────────┐
│  Llama 3.2 1B 前向推理          │
│  (use_cache=True)               │
└──────────────┬──────────────────┘
               │ past_key_values
               ▼
┌─────────────────────────────────┐
│  提取 & RoPE 移除               │
│  pk[16层][1,8,sl,64]            │
│  → keys_mat [n_pos, 8192]      │
│  → vals_mat [n_pos, 8192]      │
└──────────────┬──────────────────┘
               │ [N_total, 8192]
               ▼
┌─────────────────────────────────┐
│  PCA (torch.pca_lowrank)        │
│  → V [8192, rank]  投影矩阵    │
│  → mu [8192]       均值向量    │
│  → sigma [rank]    奇异值      │
└──────────────┬──────────────────┘
               │ sigma
               ▼
┌─────────────────────────────────┐
│  DP 比特分配                    │
│  → alloc [rank]  每分量比特数   │
│  (e.g., [8,8,8,...,2,2,...,0,0])│
└─────────────────────────────────┘
```

### 5.2 压缩阶段数据流

```
KV Cache [16层 × (1,8,sl,64)]
    │
    ├── sink [16层 × (1,8,4,64)]       ──→ 原样保存
    ├── window [16层 × (1,8,128,64)]   ──→ 原样保存
    │
    └── 压缩区域 [16层 × (1,8,nc,64)]
         │
         ├── 移除 RoPE
         │
         ├── 跨层拼接 → [nc, 8192]
         │
         ├── PCA 投影: (X - mu) @ V → [nc, rank]
         │
         ├── 对每个激活分量均匀量化
         │   → header (11B) + uint16 数据
         │
         └── DEFLATE 压缩
              → compressed bytes
```

### 5.3 解压缩阶段数据流

```
compressed bytes
    │
    ├── DEFLATE 解压
    │
    ├── 解析 header + 反量化
    │   → PCA 系数 [nc, rank]
    │
    ├── 逆 PCA: D @ V.T + mu → [nc, 8192]
    │
    ├── 切分各层: [nc, 512] → [nc, 8, 64]
    │
    ├── 重新应用 RoPE
    │
    └── 拼接 sink + 解压区域 + window
         → 恢复的 KV Cache [16层 × (1,8,sl,64)]
```

---

## 6. 关键算法细节

### 6.1 压缩比计算

```
总体压缩比 = 原始 KV 缓存大小 / (压缩区域 + sink 区域 + window 区域)

区域压缩比 = (nc × 8192 × 2 × elem_size) / 压缩后字节数
             仅衡量压缩区域的压缩效果

DEFLATE 额外压缩 = 量化后原始字节数 / DEFLATE 后字节数
```

随着序列长度增长，固定大小的未压缩区域（sink 4 + window 128 = 132 tokens）占比越来越小，总体压缩比趋近于区域压缩比。

### 6.2 比特预算计算

```python
budget = total_features * 16 / target_cr
# total_features = 8192 (跨层维度)
# 16 = 原始 BF16 比特数
# target_cr = 目标压缩比 (e.g., 16)
# budget = 8192 * 16 / 16 = 8192 bits per token position
```

这意味着在 16× 压缩目标下，每个 token 位置仅有 8192 bits（1024 字节）的预算，而原始需要 8192 × 16 = 131072 bits（16384 字节）。

### 6.3 PCA 低秩近似质量

PCA 的有效性取决于 KV 缓存数据中跨层/跨头相关性的强度。奇异值 `sigma` 的衰减速度决定了压缩上限：
- 快速衰减 → 少数主成分即可捕获大部分方差 → 高压缩比
- 缓慢衰减 → 需要更多主成分 → 压缩比受限

### 6.4 量化精度与误差关系

```
量化误差 ≈ variance / 4^bits

bits=0: 误差 = 100% (丢弃)
bits=1: 误差 = 25%
bits=2: 误差 = 6.25%
bits=4: 误差 = 0.39%
bits=8: 误差 = 0.0015%
```

DP 算法的核心思想是：将宝贵的比特预算分配给方差最大（最重要）的主成分，而完全丢弃方差接近零的尾部成分。

---

## 7. 实验结果

### 7.1 Llama 3.2 1B-Instruct, 862 tokens, CR=16×

| 指标 | CPU (zlib) | GPU (nvCOMP) |
|---|---|---|
| 压缩前 KV 缓存 | 26.94 MiB | 26.94 MiB |
| 压缩后 KV 缓存 | 5.78 MiB | 6.01 MiB |
| 节省空间 | 78.5% | 77.7% |
| Key 余弦相似度 | 0.9904 | 0.9904 |
| Value 余弦相似度 | 0.8636 | 0.8636 |
| 压缩时间 | 835 ms | 779 ms |
| 解压缩时间 | 347 ms | 496 ms |
| DEFLATE 后端 | zlib CPU | nvCOMP GPU |

**关键观察：**
- Key 的余弦相似度 (0.9904) 显著高于 Value (0.8636)，说明 Key 的跨层相关性更强
- 量化和 PCA 投影质量对 Key 和 Value 的影响不同
- nvCOMP GPU DEFLATE 在压缩速度上略优，但在解压缩上反而更慢（可能是 GPU/CPU 间数据传输开销）

### 7.2 kvtc 的适用场景

| 场景 | 说明 |
|---|---|
| **多轮对话** | 用户输入间隔期间压缩缓存，发送消息时解压 |
| **RAG 共享上下文** | 一次性压缩文档上下文，按需解压回答问题 |
| **分离式推理** | 在 prefill 和 decode 节点间传输压缩缓存 |
| **缓存分层** | 将压缩缓存保留在 CPU RAM/SSD 而非从 HBM 中驱逐 |

**性能拐点：** 对于短上下文 (<500 tokens) + 快速 GPU，重计算比解压更快。交叉点在 **2K-4K+ tokens** 左右（8B 模型），更大模型因 prefill 成本更高而更早受益。

---

## 8. 当前 PoC 的局限性

| 方面 | PoC 现状 | 论文/生产级方案 |
|---|---|---|
| DP 算法 | 贪心 DP（按效率排序分配） | 完整动态规划 |
| 校准数据 | 合成的 16 段固定文本重复使用 | 160K+ tokens 来自多样化语料 |
| vLLM 集成 | 无 | 需要 KV Connector 或 LMCache 后端 |
| 批处理 | 仅单序列 | 批量压缩/解压缩 |
| 比特选项 | 最高 8 bits | 论文 Figure 6 显示可达 30 bits |
| PCA 精度 | float32 | 可能使用混合精度加速 |

---

## 9. 代码质量评估

### 9.1 优点
- **清晰的模块化**: 校准器、压缩器、评估函数分离良好
- **向量化操作**: RoPE 处理使用批量操作，避免逐头循环
- **渐进式复杂度**: 三个文件从简单到复杂，便于理解
- **详细的输出报告**: 包含多维度的压缩和质量指标
- **灵活的配置**: 通过 `KVTCConfig` dataclass 和 CLI 参数控制

### 9.2 可改进之处
- 三个文件之间存在大量代码重复（RoPE、校准器、压缩器）
- `kvtc_rag_poc.py` 第 600 行有格式化字符串语法错误（`faster:>0`）
- 缺少单元测试
- 缺少依赖管理文件 (requirements.txt 或 pyproject.toml)

---

## 10. 总结

kvtc-poc 项目成功演示了将经典媒体压缩技术（变换编码）应用于 LLM KV 缓存压缩的可行性。其核心创新在于：

1. **跨层 PCA** — 利用注意力层间的相关性，将 16 层 × 8 头 × 64 维 = 8192 维向量投影到紧凑空间
2. **RoPE 感知** — 压缩前移除位置编码以保留低秩结构，解压后恢复
3. **自适应比特分配** — DP 算法根据方差自动决定每个主成分的量化精度
4. **两阶段压缩** — 量化（有损）+ DEFLATE（无损）的组合实现高压缩比

该方法在 16× 压缩比下实现了 Key 余弦相似度 0.99、Value 余弦相似度 0.86 的重建质量，为 LLM 推理系统中的 KV 缓存存储和传输提供了有效的压缩方案。
