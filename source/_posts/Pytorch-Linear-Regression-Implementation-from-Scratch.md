---
title: Pytorch-线性回归的从零开始实现
date: 2022-01-20 23:47:12
tags: 学习笔记
categories: 深度学习
math: true
---

[参考教程：DIVE INTO DEEP LEARNING](https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression-scratch.html)

本文章是作为一新手，对李沐大神教材的复现。加入了一些自己的见解。

## 0. 导包

``` python
import random
import torch
from d2l import torch as d2l
```

[导入`d2l`包的教程——from CSDN](https://blog.csdn.net/scar2016/article/details/115053959)

## 1. 生成数据集

为了简单起见，我们将根据带有噪声的线性模型构造一个人造数据集。 我们的任务是使用这个有限样本的数据集来恢复这个模型的参数。 我们将使用低维数据，这样可以很容易地将其可视化。 在下面的代码中，我们生成一个包含1000个样本的数据集， 每个样本包含从标准正态分布中采样的2个特征。 我们的合成数据集是一个矩阵 $\mathbf{X} \in \mathbb{R}^{1000 \times 2}$ 。

我们使用线性模型参数 $\mathbf{w}=[2,-3.4]^{\top}, b=4.2$ 和噪声项 $\epsilon$ 生成数据集及其标签：
$$
\mathbf{y}=\mathbf{X} \mathbf{w}+b+\epsilon
$$
你可以将ϵϵ视为模型预测和标签时的潜在观测误差。 在这里我们认为标准假设成立，即ϵϵ服从均值为0的正态分布。 为了简化问题，我们将标准差设为0.01。 下面的代码生成合成数据集。

``` python
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b  # 返回矩阵向量积
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))  # 返回一个列数为1的张量（行数由python解释器决定）

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)  # 生成一个含1000个数据的数据集

print('features:', features[0], '\nlabel:', labels[0])  # Example
```

通过生成第二个特征`features[:, 1]`和`labels`的散点图， 可以直观观察到两者之间的线性关系。

``` python
# 散点图
d2l.set_figsize()  # 设置图片的大小
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()
```

![散点图](image-20220121001356955.png)

## 2. 读取数据集

回想一下，训练模型时要对数据集进行遍历，**每次抽取一小批量样本**，并使用它们来更新我们的模型。 由于这个过程是训练机器学习算法的基础，所以有必要定义一个函数， 该函数能**打乱数据集中的样本并以小批量方式获取数据**。

在下面的代码中，我们定义一个`data_iter`函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为`batch_size`的小批量。 每个小批量包含一组特征和标签。

``` python
"""
     该函数接收批量大小、特征矩阵和标签向量作为输入，
     生成大小为batch_size的小批量。
     每个小批量包含一组特征和标签。
"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 返回的是行数
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)  # random.shuffle()--将序列的所有元素随机排序
    for i in range(0, num_examples, batch_size):  # range(start, stop[, step])
        batch_indices = np.array(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

        
"""
    读取数据集
"""
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```



### 补充：

#### `shuffle()`函数

将序列的所有元素随机排序。

用法：

``` python
import random

random.shuffle (lst )
```



#### `Python3 range()`函数

两种用法：

``` python
range(stop)
range(start, stop[, step])
```



#### `yield`用法

以一个斐波那契数列的打印算法来举例

``` python
#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b      # 使用 yield
        # print b 
        a, b = b, a + b 
        n = n + 1
 
for n in fab(5): 
    print n
```

打印结果为：

```
1
1
2
3
5
```

简单地讲，yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，调用 fab(5) 不会执行 fab 函数，而是返回一个 iterable 对象！在 for 循环执行时，每次循环都会执行 fab 函数内部的代码，执行到 yield b 时，fab 函数就返回一个迭代值，下次迭代时，代码从 yield b 的下一条语句继续执行，而函数的本地变量看起来和上次中断执行前是完全一样的，于是函数继续执行，直到再次遇到 yield。

也可以手动调用 fab(5) 的 next() 方法（因为 fab(5) 是一个 generator 对象，该对象具有 next() 方法），这样我们就可以更清楚地看到 fab 的执行流程：
