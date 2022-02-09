---
title: PyTorch下的自动求导
date: 2022-02-09 21:27:16
tags: 学习笔记
categories: 深度学习
math: true
---





在学习`PyTorch`框架下的自动求导（`autograd`）功能，首先要对矩阵求导有一个了解



## 矩阵求导的本质

这位答主写的非常到位了，直接贴文章链接：[矩阵求导的本质与分子布局、分母布局的本质](https://zhuanlan.zhihu.com/p/263777564)，以及[矩阵求导公式的数学推导](https://zhuanlan.zhihu.com/p/273729929)

以下仅仅是记一下结论：

### 一、向量变元的实值标量函数

$$
f(\pmb{x}),\pmb{x}=[x_1,x_2,\cdots,x_n]^T
$$

#### 1. 四个法则

1.1 常数求导
$$
\frac{\partial c}{ \partial \pmb{x}}=\pmb{0}_{n \times 1}
$$
1.2 线性法则
$$
\frac{\partial{[c_1f(\pmb{x})+c_2g(\pmb{x})]}}{\partial{\pmb{x}}} =  c_1\frac{\partial f(\pmb{x})}{\partial{\pmb{x}}} + c_2\frac{\partial g(\pmb{x})}{\partial{\pmb{x}}}
$$
1.3 乘积法则
$$
\frac{\partial{[f(\pmb{x})g(\pmb{x})]}}{\partial{\pmb{x}}} =  \frac{\partial f(\pmb{x})}{\partial{\pmb{x}}}g(\pmb{x}) +f(\pmb{x})\frac{\partial g(\pmb{x})}{\partial{\pmb{x}}}
$$
1.4 商法则
$$
\frac{\partial{\left[\frac{f(\pmb{x})}{g(\pmb{x})}\right]}}{\partial{\pmb{x}}} =  \frac{1}{g^2(\pmb{x})}\left[ \frac{\partial f(\pmb{x})}{\partial{\pmb{x}}}g(\pmb{x}) -f(\pmb{x})\frac{\partial g(\pmb{x})}{\partial{\pmb{x}}}   \right]
$$
