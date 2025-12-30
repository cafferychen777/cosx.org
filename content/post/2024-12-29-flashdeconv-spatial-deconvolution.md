---
title: "FlashDeconv：用随机素描实现百万级空间转录组去卷积"
date: "2024-12-29"
author: "Chen Yang"
categories:
  - 统计计算
  - 机器学习
  - 生物信息
tags:
  - 空间转录组学
  - 去卷积
  - 随机算法
  - 数值线性代数
  - Python
slug: flashdeconv-spatial-deconvolution
---

> 本文介绍我们开发的空间转录组去卷积工具 FlashDeconv，它通过结构保持的随机素描(randomized sketching)实现了 O(N) 的时间复杂度，能在 3 分钟内完成百万级数据点的分析。

## 空间转录组学：从"在哪里"到"有多少"

近年来，空间转录组学技术飞速发展，从最初的 10x Visium 到如今的 Visium HD、Stereo-seq、Xenium 等亚细胞分辨率平台，我们获得了前所未有的能力来研究组织中基因表达的空间分布。然而，一个核心挑战始终存在：如何从每个空间位点的混合表达谱中推断出各种细胞类型的比例？

这个问题被称为**空间去卷积(spatial deconvolution)**。想象你有一杯混合果汁，你需要推断出苹果汁、橙汁、葡萄汁各占多少比例——只不过在这里，"果汁"是基因表达谱，"比例"是细胞类型组成。

现有的去卷积方法，如 Cell2Location 和 RCTD，虽然在准确性上表现优异，但计算效率令人头疼。Cell2Location 使用贝叶斯推断，需要 GPU 加速；RCTD 需要对每个位点独立求解，计算量随数据规模线性增长——但常数因子很大。当面对 Visium HD 等新一代平台产生的百万级位点数据时，这些方法要么需要昂贵的 GPU 集群，要么需要数小时甚至数天的运行时间。

我们开发的 [FlashDeconv](https://github.com/cafferychen777/flashdeconv) 直面这一挑战。核心思想是：**通过随机素描压缩基因空间，同时保持细胞类型的可区分性**。

## Talk is cheap. Show me the numbers.

在深入算法细节之前，让我们先看看 FlashDeconv 的实际表现。使用一台普通的 MacBook Pro（M2 Max，32GB 内存，无 GPU），我们测试了不同规模的数据：

| 数据规模 | 运行时间 | 内存占用 |
|:---------|:---------|:---------|
| 10K 位点 | < 1 秒 | < 1 GB |
| 100K 位点 | ~4 秒 | ~2 GB |
| 1M 位点 | ~3 分钟 | ~21 GB |

这是什么概念？百万位点的 Visium HD 数据，3 分钟就能完成分析，而且不需要 GPU。在准确性方面，FlashDeconv 在 Spotless 基准测试的 56 个合成数据集上取得了 0.944 的平均 Pearson 相关系数，与 Cell2Location 等顶级方法相当。

更重要的是，FlashDeconv 在**稀有细胞类型检测**上表现出色。传统方法通常基于方差选择基因，这会系统性地忽略低丰度细胞类型（如肿瘤中的免疫细胞、肝脏中的 Tuft 细胞）。FlashDeconv 使用杠杆分数(leverage score)加权采样，能够有效保留这些稀有但重要的细胞信号。

## 算法原理：随机素描遇见图正则化

FlashDeconv 的核心是将空间去卷积问题转化为**图正则化的非负最小二乘问题**，并在压缩的"素描空间"中求解：

`$$
\min_{\beta \geq 0} \frac{1}{2}\|S(Y - X\beta)\|_F^2 + \frac{\lambda}{2}\text{tr}(\beta^T L \beta) + \rho\|\beta\|_1
$$`

其中 `$Y$` 是空间表达矩阵（位点×基因），`$X$` 是参考细胞类型特征矩阵，`$\beta$` 是我们要估计的细胞类型丰度，`$S$` 是随机素描矩阵，`$L$` 是空间图的拉普拉斯矩阵。

### 第一步：Log-CPM 标准化与杠杆分数基因选择

预处理的目标是稳定方差并选择信息量最大的基因。我们使用 Log-CPM（counts per million 的对数变换）而非流行的 Pearson 残差：

`$$
Y_{\text{norm}} = \log_2\left(\frac{Y}{\text{lib\_size}} \times 10^6 + 1\right)
$$`

为什么不用 Pearson 残差？因为 Pearson 残差虽然在单细胞聚类中表现良好，但它假设基因表达服从负二项分布并按期望值标准化，这会放大低表达基因的噪声。对于去卷积任务，我们需要保持细胞类型特征矩阵 `$X$` 和空间表达 `$Y$` 之间的一致性，Log-CPM 是更稳健的选择。

基因选择是另一个关键环节。传统的高变异基因(HVG)选择方法偏向于高表达的管家基因——它们的方差大，但对区分细胞类型帮助不大。FlashDeconv 使用**杠杆分数(leverage score)**来量化每个基因的"信息独特性"：

`$$
\ell_j = \|U_j\|_2^2
$$`

其中 `$U$` 是细胞类型特征矩阵 `$X$` 的左奇异向量矩阵。杠杆分数高的基因对应于在低维主成分空间中占据独特方向的基因——这些通常是稀有细胞类型的标记基因。

### 第二步：结构保持的随机素描

空间转录组数据通常有 20000+ 个基因，直接求解计算量巨大。FlashDeconv 使用 **CountSketch** 将基因空间压缩到 512 维：

```python
# CountSketch 的核心思想
for gene_j in genes:
    bucket = hash(gene_j) % sketch_dim  # 随机分桶
    sign = random_sign(gene_j)          # 随机符号
    sketch[:, bucket] += sign * data[:, gene_j]
```

CountSketch 的数学保证来自 Johnson-Lindenstrauss 引理：以高概率，压缩后向量之间的欧氏距离得到保持。关键创新是**杠杆分数加权采样**：在素描过程中，根据杠杆分数对基因进行重要性采样，使稀有细胞类型的标记基因获得更大的权重，避免在哈希碰撞中被高丰度细胞类型的信号淹没。

### 第三步：稀疏图拉普拉斯正则化

空间去卷积的一个核心假设是**空间平滑性**：相邻位点的细胞类型组成应该相似。传统方法如 CARD 使用高斯核构建全连接的相似度矩阵，这导致 `$O(N^2)$` 的内存复杂度——百万位点需要 TB 级内存，显然不可行。

FlashDeconv 使用稀疏的 k-NN 图。对于每个位点，只记录其 k 个最近邻（默认 k=6），这给出 `$O(Nk)$` 的复杂度。图拉普拉斯矩阵定义为：

`$$
L = D - W
$$`

其中 `$W$` 是邻接矩阵，`$D$` 是度矩阵。正则化项 `$\text{tr}(\beta^T L \beta)$` 惩罚相邻位点估计值的差异，等价于高斯马尔可夫随机场(GMRF)的精度矩阵。

### 优化：Numba 加速的块坐标下降

最终的优化问题通过**块坐标下降(Block Coordinate Descent, BCD)**求解。每次更新一个位点的所有细胞类型比例：

`$$
\beta_i^{\text{new}} = \text{proj}_{\geq 0}\left(\beta_i - \frac{1}{\eta_i}\nabla_i f(\beta)\right)
$$`

其中投影到非负象限确保比例的物理意义。由于图拉普拉斯是稀疏的，每次更新只涉及当前位点及其邻居，复杂度为 `$O(k \cdot n_{\text{cell\_types}})$`。

整个算法使用 Numba JIT 编译加速，避免了 Python 循环的开销。配合稀疏矩阵运算，实现了百万级数据的高效处理。

## 快速上手

FlashDeconv 提供了与 Scanpy 无缝集成的 API：

```python
import scanpy as sc
import flashdeconv as fd

# 加载数据
adata_st = sc.read_h5ad("visium_hd.h5ad")
adata_ref = sc.read_h5ad("reference.h5ad")

# 一行完成去卷积
fd.tl.deconvolve(adata_st, adata_ref, cell_type_key="cell_type")

# 结果自动存储在 adata_st 中
adata_st.obsm["flashdeconv"]          # 细胞类型比例
adata_st.obs["flashdeconv_dominant"]  # 主导细胞类型

# 可视化
sc.pl.spatial(adata_st, color="flashdeconv_Hepatocyte")
```

安装只需一行命令：

```bash
pip install flashdeconv
```

## 调参建议：`lambda_spatial` 的选择

FlashDeconv 的关键超参数是 `lambda_spatial`，控制空间正则化的强度。根据平台的位点大小和稀疏程度，我们建议：

| 平台 | 位点大小 | 推荐 `lambda_spatial` |
|:-----|:---------|:---------------------|
| Standard Visium | 55µm | 1000–10000 |
| Visium HD (16µm) | 16µm | 5000–20000 |
| Visium HD (8µm) | 8µm | 10000–50000 |
| Visium HD (2µm) | 2µm | 50000–100000 |
| Stereo-seq | 0.5–1µm | 50000–200000 |

直觉是：**位点越小，表达越稀疏，需要越多地依赖空间邻居的信息**。如果结果看起来像"椒盐噪声"，增大 `lambda_spatial`；如果过度模糊，减小它。

## 总结

FlashDeconv 通过三个核心创新实现了高效的空间去卷积：

1. **杠杆分数加权的基因选择**：保留稀有细胞类型的信号
2. **结构保持的随机素描**：将基因空间压缩 40 倍而不损失关键信息
3. **稀疏图拉普拉斯正则化**：以 `$O(N)$` 复杂度实现空间平滑

我们希望 FlashDeconv 能帮助研究者更高效地分析空间转录组数据，特别是面对 Visium HD 等新一代平台带来的海量数据挑战。工具已开源在 [GitHub](https://github.com/cafferychen777/flashdeconv)，欢迎使用和反馈！

### 引用

如果 FlashDeconv 对您的研究有帮助，请引用：

> Yang, C., Chen, J. & Zhang, X. FlashDeconv enables atlas-scale, multi-resolution spatial deconvolution via structure-preserving sketching. *bioRxiv* (2025). https://doi.org/10.64898/2025.12.22.696108
