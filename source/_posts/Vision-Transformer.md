---
title: Vision-Transformer
date: 2022-04-13 01:50:04
tags: 论文阅读
categories: 深度学习
math: true
---

本文将以模型解释与代码结合起来，来向大家解释Vision Transformer Model(ViT)



论文地址如下：

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

[论文代码](https://github.com/google-research/vision_transformer)



## 论文模型总览

我们将图像分割成固定大小的块，线性嵌入每个块，添加位置嵌入，并将生成的向量序列馈送到标准的`Transformer`编码器。为了执行分类，我们使用向序列添加额外可学习的“分类标记”的标准方法。

![ViT模型总览](Vision-Transformer/Model-Overview.png)



## 分块与降维

对于图像来讲，一般是三通道（BGR256）的彩色图片，但是我们想用现成的`Transformer`模型对图像进行处理，那么我们就应该对图像进行降维处理。

首先把$\mathbf{x}_{p} \in \mathbb{R}^{H \times W \times C}$的图像，变成一个$\mathbf{x}_{p} \in \mathbb{R}^{N \times\left(P^{2} \cdot C\right)}$的**sequence of flattened 2D patches**。其可视为一系列的展平的2D块的序列，这个序列中一共有$N = \frac{ HW }{P^{2}}$个展平的2D块，$N$即为`Transformer`输入的`sequence`的长度。其中每个块的维度是$\left(P^2 \cdot C \right)$，其中$H$和$W$是图像的高和宽，$P$是块大小，$C$是图片的通道数。

那么这一步在代码种是怎么做的呢？

``` python
x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
```

[einops：优雅地操作张量维度](https://zhuanlan.zhihu.com/p/342675997)
