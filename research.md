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

#### 为什么需要校准？

PCA 压缩不是一种"通用"压缩（如 gzip），它依赖于数据的**统计结构**。校准阶段的核心目的是从该模型的 KV 缓存样本中学习这些统计量，具体产出三组参数：

| 校准产出 | 变量 | 形状 | 作用 |
|---|---|---|---|
| **PCA 投影矩阵** | `V_k`, `V_v` | `[8192, rank]` | 定义主成分方向——即数据方差最大的 `rank` 个正交方向。压缩时沿这些方向投影（降维），解压时沿反方向投影（恢复）。**这就是校准要"找到"的投影方向维度** |
| **均值向量** | `mu_k`, `mu_v` | `[8192]` | PCA 要求数据先中心化（减去均值），均值需要从校准样本中估计。解压时将均值加回 |
| **比特分配方案** | `alloc_k`, `alloc_v` | `[rank]` | 基于校准中得到的奇异值 `S`（反映每个主成分的方差/重要性），DP 算法决定每个主成分应分配多少比特。方差大的成分（信息量大）分配更多比特，方差极小的成分直接丢弃（0 比特） |

**关键理解**：校准是"离线一次性"的。对同一个模型（如 Llama-3.2-1B），只需校准一次，产出的 `V`、`mu`、`alloc` 可以复用于该模型所有后续的 KV 缓存压缩/解压。不同的输入文本会产生不同的 KV 缓存值，但它们的统计结构（主成分方向、方差分布）由模型权重决定，在不同输入间高度稳定。

**如果不校准会怎样？** 没有 `V`（投影方向），就无法进行 PCA 降维；没有 `mu`（均值），中心化无法正确执行；没有 `alloc`（比特分配），就不知道哪些主成分重要、该保留多少精度。三者缺一不可。

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

#### 2.5.1 `torch.pca_lowrank` 函数详解

`torch.pca_lowrank` 是 PyTorch 提供的**低秩近似 PCA** 实现，基于随机化 SVD（Randomized SVD）算法。它不计算完整的奇异值分解，而是高效地近似出前 `q` 个最大的奇异值和对应的奇异向量。

**函数签名：**

```python
torch.pca_lowrank(A, q=None, center=True, niter=2)
```

**参数说明：**

| 参数 | 类型 | 说明 |
|---|---|---|
| `A` | Tensor | 输入矩阵，形状 `[n, p]`。在 kvtc 中即中心化后的 KV 缓存矩阵 `[n_pos, 8192]` |
| `q` | int | 要求的近似秩（保留的主成分数量）。默认为 `min(6, n, p)`。kvtc 中设为 `min(rank, n, p)`，rank 默认 4096 |
| `center` | bool | 是否自动中心化。kvtc 中手动中心化后传入，此处未显式设置（默认 True，但已中心化的数据再中心化不影响结果） |
| `niter` | int | 幂迭代（power iteration）次数，用于提高近似精度。kvtc 中设为 5（默认为 2） |

**返回值：**

```python
U, S, V = torch.pca_lowrank(centered, q=rank, niter=5)
# U: [n, q]    — 左奇异向量（每行是一个样本在主成分空间的坐标）
# S: [q]       — 奇异值（降序排列，反映每个主成分的重要性）
# V: [p, q]    — 右奇异向量（每列是一个主成分方向）
```

关系满足：`A ≈ U · diag(S) · V^T`

**内部算法 — 随机化 SVD：**

`pca_lowrank` 底层调用 `torch.svd_lowrank`，其核心是 Halko-Martinsson-Tropp (2011) 随机化算法：

```
算法流程（简化版）：

输入: A [n×p], 目标秩 q, 幂迭代次数 niter

1. 随机投影（Range Finding）:
   Ω = randn(p, q)           # 生成随机高斯矩阵
   Y = A @ Ω                  # 将 A 投影到 q 维随机子空间 [n×q]

2. 幂迭代（Power Iteration, 重复 niter 次）:
   for i in 1..niter:
       Y = A @ (A^T @ Y)      # 等价于 (AA^T)^niter @ A @ Ω
                               # 增强主奇异向量，抑制噪声方向

3. 正交化:
   Q, _ = QR(Y)               # Q [n×q] 是 A 的列空间的正交基

4. 小矩阵 SVD:
   B = Q^T @ A                # 投影到低维 [q×p]
   Û, S, V = SVD(B)           # 对小矩阵做精确 SVD

5. 恢复:
   U = Q @ Û                  # 左奇异向量 [n×q]

输出: U [n×q], S [q], V [p×q]
```

**为什么用随机化 SVD 而不是精确 SVD？**

| 对比维度 | 精确 SVD (`torch.linalg.svd`) | 随机化 SVD (`pca_lowrank`) |
|---|---|---|
| 时间复杂度 | O(n·p·min(n,p)) | O(n·p·q + q²·(n+p)) |
| kvtc 场景 (n=1000, p=8192, q=4096) | O(8.2×10⁹) | O(3.4×10¹⁰) — 但常数更小且 GPU 友好 |
| 内存 | 需要完整 U [n×p] | 只需 U [n×q] |
| 精度 | 精确 | 近似（niter 越大越精确） |
| GPU 加速 | 受限（内部 LAPACK） | 全程矩阵乘法，GPU 高度并行 |

在 kvtc 的典型场景中 `p=8192, q≤4096`，随机化 SVD 的优势在于：
1. **只计算需要的前 q 个主成分**，不浪费计算在尾部奇异值上
2. **全程使用矩阵乘法**（GEMM），GPU 上高度优化
3. **内存效率**：不需要存储完整的 `[n×8192]` U 矩阵

**`niter=5` 的作用：**

幂迭代次数控制近似质量。每次迭代将矩阵乘以 `(AA^T)`，效果是将奇异值谱中的间隙放大 — 第 i 个奇异值被放大为 `σᵢ^(2·niter+1)`，使前 q 个奇异向量更容易被捕获。

```
niter=0: 粗略近似，可能遗漏中等大小的奇异值
niter=2: 默认值，大多数场景足够
niter=5: kvtc 的选择，确保 PCA 基的高精度（校准是离线过程，多几次迭代的开销可接受）
```

**在 kvtc 中的具体使用：**

```python
# kvtc_poc.py 第 151-157 行
def pca(self, data, rank):
    mu = data.mean(dim=0)                    # [8192] 均值向量
    centered = data - mu.unsqueeze(0)        # [n_pos, 8192] 中心化
    n, p = centered.shape                    # n=n_pos, p=8192
    r = min(rank, n, p)                      # 实际秩 = min(4096, n_pos, 8192)
    U, S, V = torch.pca_lowrank(centered, q=r, niter=5)
    return V, mu, S
    # V: [8192, r] — 投影矩阵，每列是一个主成分方向
    # mu: [8192]   — 均值，解压时需要加回
    # S: [r]       — 奇异值，传给 DP 比特分配算法
```

压缩时使用 `V` 投影：`projected = (data - mu) @ V`，将 `[n_pos, 8192]` 压缩到 `[n_pos, r]`。
解压时使用 `V^T` 反投影：`reconstructed = projected @ V^T + mu`，恢复到 `[n_pos, 8192]`。

`S`（奇异值）的平方 `S²` 正比于每个主成分解释的方差，直接输入 DP 比特分配算法，方差大的成分获得更多量化比特。

### 2.6 DP 比特分配 (Dynamic Programming Bit Allocation)

这是 kvtc 的关键创新之一。DP 算法决定 PCA 投影后的每组分量应当分配多少比特来量化，核心目标是：**在固定的比特预算内，最小化量化引入的 Frobenius 重建误差**。

#### 2.6.1 函数签名与输入参数

```python
def dp_alloc(self, sigma, total_features, label=""):
```

调用位置（`kvtc_poc.py` 第 306-307 行）：

```python
self.alloc_k = self.dp_alloc(self.sigma_k, CROSS_LAYER_DIM, "Keys")
self.alloc_v = self.dp_alloc(self.sigma_v, CROSS_LAYER_DIM, "Values")
```

**参数详解：**

| 参数 | 类型 | 含义 | 实际取值 |
|---|---|---|---|
| `sigma` | Tensor `[r]` | PCA 奇异值向量，由 `torch.pca_lowrank` 返回的 `S`。`sigma[i]²` 正比于第 i 个主成分解释的方差，降序排列。前面的成分方差大（信息量大），后面的成分方差小（可丢弃） | 长度 = `min(pca_rank, n_pos, 8192)`，通常为 4096 |
| `total_features` | int | **跨层拼接后的特征维度**，即 `CROSS_LAYER_DIM = N_LAYERS × N_KV_HEADS × HEAD_DIM = 16 × 8 × 64 = 8192`。它代表原始 KV 缓存中每个 token 位置的总特征数。用于计算比特预算：压缩前每个 token 需要 `total_features × 16` 比特（FP16，每元素 16 位），压缩后的比特预算 = 压缩前比特数 / 目标压缩比 | 8192 |
| `label` | str | 仅用于日志打印的标签（"Keys" 或 "Values"） | "Keys" / "Values" |

此外还使用了配置参数 `self.cfg.target_cr`：

| 配置项 | 类型 | 含义 | 默认值 |
|---|---|---|---|
| `target_cr` | int | **目标压缩比**（Compression Ratio）。表示压缩后数据量是原始数据的 1/target_cr。例如 `target_cr=16` 意味着压缩到原始大小的 1/16（即 16× 压缩） | 16 |

#### 2.6.2 比特预算计算

```python
budget = int(total_features * 16 / self.cfg.target_cr)
```

推导过程：
- 压缩前：每个 token 位置有 `total_features = 8192` 个 FP16 值，每个 16 比特 → 共 `8192 × 16 = 131072` 比特
- 目标压缩比 `target_cr = 16`
- 压缩后比特预算：`131072 / 16 = 8192` 比特/token

这意味着 DP 算法需要将 4096 个 PCA 分量的量化总开销控制在 8192 比特以内（理论平均 2 比特/分量，但实际因每组 OVERHEAD=32 比特的固定开销，有效可用比特更少；大部分尾部分量会被分配 0 比特直接丢弃）。

#### 2.6.3 算法常量

```python
OVERHEAD = 32       # 每个激活组的固定开销（比特），用于存储缩放/偏移量化参数
GROUP_SIZES = [16, 64, 256, 1024]   # 候选分组大小
BIT_OPTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 每元素候选比特数
```

| 常量 | 含义 |
|---|---|
| `OVERHEAD = 32` | 每激活一个分组，需额外消耗 32 比特存储该组的量化参数（vmin + scale）。若组的比特数为 0（丢弃），则无此开销 |
| `GROUP_SIZES` | 将 PCA 分量按连续块分组的候选大小。例如 `gs=64` 表示每 64 个相邻分量共享同一个比特数。较大的组能更好地**分摊 OVERHEAD 开销**（32 比特 / 1024 分量 = 0.03 比特/分量），但灵活性降低 |
| `BIT_OPTIONS` | 每个分量可分配 0-8 比特。0 比特 = 完全丢弃该组（不存储数据，也不产生 OVERHEAD 开销）；8 比特 = 最高精度量化（255 个量化级别） |

#### 2.6.4 量化误差模型

```python
var = (sigma ** 2).cpu().numpy()   # 每个分量的方差

def quant_error(group_var, bits):
    if bits == 0:
        return group_var              # 丢弃：误差 = 全部方差
    return group_var / (4.0 ** bits)  # 均匀量化：误差 ≈ var / 4^b
```

- `group_var`：一组分量的**累计方差**（= 该组内所有 `sigma[i]²` 之和）
- `bits = 0`：完全丢弃，重建误差等于全部方差
- `bits = b`（b > 0）：均匀量化后残差误差近似为 `var / 4^b`。这是因为 b 比特量化有 `2^b` 个级别，量化步长 ∝ `1/2^b`，量化噪声方差 ∝ `(步长)² ∝ 1/4^b`
- 每多 1 比特，误差减少 4 倍（6 dB）

#### 2.6.5 贪心背包算法流程

算法分两层循环：外层遍历所有候选分组大小，内层对每种分组大小执行贪心背包，最终选择**总重建误差最小**的方案。

```
输入:
  sigma[0..n-1]   — PCA 奇异值（降序）
  budget          — 比特预算（例: 8192）

对每种 group_size (gs) ∈ {16, 64, 256, 1024}:
  1. 分组: 将 n 个分量按顺序切成 ⌊n/gs⌋ 个完整组
     例: n=4096, gs=64 → 64 个组，每组 64 个分量
     注: 若 n 不能被 gs 整除，剩余分量不参与分配（等同于 0 比特丢弃）
     
  2. 计算每组方差:
     g_var[g] = Σ sigma[g*gs .. (g+1)*gs-1]²
     （第 g 组内所有分量的方差之和）

  3. 生成候选操作:
     对每个组 g 和比特等级 b ∈ {1,2,...,8}:
       成本 cost = OVERHEAD + gs × b      （32 + 64×8 = 544 比特）
       增益 gain = g_var[g] × (1 - 1/4^b) （比丢弃减少的误差）
       效率 eff  = gain / cost             （单位比特的误差减少量）

  4. 按效率降序排列所有候选操作

  5. 贪心分配:
     遍历排序后的候选，依次尝试：
     - 若该组尚未分配且 cost ≤ 剩余预算 → 分配，扣减预算
     - 若该组已分配低比特 → 计算升级的增量成本与增量增益，
       若增量效率仍划算且预算允许 → 升级

  6. 计算该分组大小下的总重建误差

选择总误差最小的 group_size 方案 → 输出 alloc[0..n-1]
```

**输出：** `alloc` 数组，长度 = PCA 分量数。`alloc[i]` = 第 i 个分量分配的比特数。

#### 2.6.6 典型分配结果

以 `target_cr=16, total_features=8192, pca_rank=4096` 为例：

```
budget = 8192 × 16 / 16 = 8192 比特/token

典型分配模式（group_size=64 被选中时）：
  分量 0-63:    8 比特  — 方差最大的主成分，保留最高精度
  分量 64-127:  6 比特  — 次重要的成分
  分量 128-255: 4 比特  — 中等重要性
  分量 256-511: 2 比特  — 低重要性
  分量 512+:    0 比特  — 方差极小，完全丢弃
```

这符合论文 Figure 6 的描述：领先的主成分获得高比特，尾部成分被丢弃。由于 PCA 已按方差降序排列分量，这种阶梯式分配是最优策略的近似。

#### 2.6.7 与论文的差异

PoC 实现使用**贪心背包**而非论文中的**完整 DP**（动态规划表）。完整 DP 可以找到全局最优分配，但状态空间为 O(n × budget)，对 n=4096、budget=8192 会产生约 3300 万个状态。贪心方法牺牲了少量最优性，但运行速度快且结果接近最优。

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

## 9. 与量化压缩、线性 Attention 的对比及复用分析

### 9.1 三种 KV 缓存压缩范式对比

| 维度 | kvtc (变换编码) | 量化压缩 (FP8/KIVI/KVQuant) | 线性 Attention (Linear Attention) |
|---|---|---|---|
| **压缩原理** | PCA 跨层投影 + 可变比特量化 + 熵编码 | 降低每个元素的比特宽度 (16→8/4/2 bit) | 用线性核替代 softmax，将 KV 缓存压缩为固定大小的状态矩阵 |
| **压缩比** | 5-40× (可调) | 2-8× (FP8=2×, INT4=4×, INT2=8×) | 理论上无限 (O(T) → O(1)，缓存大小与序列长度无关) |
| **精度影响** | Key cos≈0.99, Value cos≈0.86 @ 16× | FP8 几乎无损; INT2 在检索任务上显著退化 | 对长距离依赖有明显精度损失，softmax 的非线性特性难以完全保留 |
| **计算开销** | 离线校准 + 在线压缩/解压 (ms级) | 在线量化/反量化 (几乎零开销) | 需要从头训练或微调模型架构 |
| **适用时机** | 存储/传输/卸载场景 (对话间隙、跨节点) | 推理全程，始终启用 | 推理全程，替代标准 attention |
| **是否需要重新训练** | 否 (仅需校准) | 否 (PTQ) 或轻微微调 | 是 (架构变更，需要预训练或蒸馏) |
| **对现有模型的兼容性** | 高 (后处理，不改模型) | 高 (后处理) | 低 (需要修改模型架构) |

### 9.2 深入对比分析

**kvtc vs 量化压缩：**

kvtc 和量化压缩在**压缩维度**上完全不同：
- **量化压缩**逐元素降低精度 — 将每个 FP16 值独立映射到更少的比特。它不利用元素间的相关性，压缩比上限由比特宽度决定 (最多 8×)。
- **kvtc** 利用**跨层/跨头的统计相关性** — 通过 PCA 发现冗余维度后直接丢弃，然后对保留的维度再做量化。这使得 kvtc 能达到远超量化的压缩比 (16-40×)。

论文中明确指出：kvtc 不替代 FP8 量化。FP8 在推理过程中零成本提供 2× 压缩，而 kvtc 专为**存储和卸载**场景设计，两者的使用时机不重叠。

**kvtc vs 线性 Attention：**

线性 Attention (如 Linear Transformer, RWKV, Mamba 等) 从**模型架构层面**解决 KV 缓存问题：
- 用线性核 `φ(Q)·φ(K)^T` 替代 `softmax(QK^T/√d)`，使 KV 缓存可压缩为固定大小的状态矩阵 `S = Σ φ(K_i)^T · V_i`
- 缓存大小与序列长度无关 — 理论上实现了"无限"压缩
- 但代价是精度损失：softmax attention 的非线性特性（尤其是尖锐的注意力分布）难以被线性核捕获

kvtc 则是**后处理方法** — 不改变模型架构，对已有的 softmax attention 模型的 KV 缓存做有损压缩。

### 9.3 复用与组合可能性

这三种方法操作在不同的层级和时机上，存在**互补**关系，可以组合使用：

**1. kvtc + 量化压缩（可组合 ✓）**

这是最自然的组合方式：

```
推理过程中:  FP16 KV Cache → FP8 量化 (2× 压缩, 零开销)
存储/卸载时: FP8 KV Cache → kvtc 压缩 (额外 8-20× 压缩)
总压缩比:    2 × 8-20 = 16-40×
```

- 量化在推理**过程中**持续生效，减少 HBM 带宽消耗
- kvtc 在需要**存储/传输**时启用，进一步压缩已量化的缓存
- 需要注意：kvtc 的 PCA 基需要在量化后的数据上重新校准，因为量化会改变数据分布
- 论文 Table 1 中已验证 kvtc 在 FP8 量化后的 KV 缓存上仍然有效

**2. kvtc + 线性 Attention（难以组合 ✗）**

```
线性 Attention: KV Cache → 固定大小状态矩阵 S [d×d]
kvtc:           标准 KV Cache [T×d] → 压缩存储
```

- 线性 Attention 已经将 KV 缓存压缩为固定大小的状态矩阵，没有"跨位置冗余"可供 PCA 利用
- kvtc 的核心假设 — KV 缓存在**token 位置维度**上有冗余 — 在线性 Attention 的状态矩阵上不成立
- 两种方法在根本上是**替代关系**而非互补关系

**3. 量化 + 线性 Attention（可组合 ✓）**

- 线性 Attention 的状态矩阵仍可使用低精度量化 (FP8/INT8)
- 这是正交的优化：一个减少缓存大小的序列依赖性，一个减少每元素的精度
- RWKV、Mamba 等模型已在实践中使用量化状态

### 9.4 组合推荐

对于生产部署，推荐的组合策略：

| 场景 | 推荐方案 | 预期压缩比 |
|---|---|---|
| 标准 Transformer + 短上下文 | FP8 量化 | 2× |
| 标准 Transformer + 长上下文 + 多轮 | FP8 + kvtc | 16-40× |
| 新模型训练 + 超长上下文 | 线性 Attention + 量化 | 缓存固定大小 |
| 混合架构 (部分层线性 + 部分层 softmax) | 线性层: 量化; softmax层: FP8 + kvtc | 视架构而定 |

---

## 10. 代码质量评估

### 10.1 优点
- **清晰的模块化**: 校准器、压缩器、评估函数分离良好
- **向量化操作**: RoPE 处理使用批量操作，避免逐头循环
- **渐进式复杂度**: 三个文件从简单到复杂，便于理解
- **详细的输出报告**: 包含多维度的压缩和质量指标
- **灵活的配置**: 通过 `KVTCConfig` dataclass 和 CLI 参数控制

### 10.2 可改进之处
- 三个文件之间存在大量代码重复（RoPE、校准器、压缩器）
- `kvtc_rag_poc.py` 第 600 行有格式化字符串语法错误（`faster:>0`）
- 缺少单元测试
- 缺少依赖管理文件 (requirements.txt 或 pyproject.toml)

---

## 11. 总结

kvtc-poc 项目成功演示了将经典媒体压缩技术（变换编码）应用于 LLM KV 缓存压缩的可行性。其核心创新在于：

1. **跨层 PCA** — 利用注意力层间的相关性，将 16 层 × 8 头 × 64 维 = 8192 维向量投影到紧凑空间
2. **RoPE 感知** — 压缩前移除位置编码以保留低秩结构，解压后恢复
3. **自适应比特分配** — DP 算法根据方差自动决定每个主成分的量化精度
4. **两阶段压缩** — 量化（有损）+ DEFLATE（无损）的组合实现高压缩比

该方法在 16× 压缩比下实现了 Key 余弦相似度 0.99、Value 余弦相似度 0.86 的重建质量，为 LLM 推理系统中的 KV 缓存存储和传输提供了有效的压缩方案。
