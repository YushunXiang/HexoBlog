---
title: Pytorch-基于梯度下降算法的线性回归模型
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



#### `yield`关键字用法

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



## 3. 初始化参数模型

在我们开始用小批量随机梯度下降优化我们的模型参数之前， 我们需要先有一些参数。 在下面的代码中，我们通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重， 并将偏置初始化为0。

``` python
"""
    初始化模型参数
"""
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

在初始化参数之后，我们的任务是更新这些参数，直到这些参数足够拟合我们的数据。 每次更新都需要计算损失函数关于模型参数的梯度。



## 4. 定义模型

接下来，我们必须定义模型，将模型的输入和参数同模型的输出关联起来。 回想一下，要计算线性模型的输出， 我们只需计算输入特征$\mathbf{X}$和模型权重$\mathbf{w}$的矩阵-向量乘法后加上偏置$b$。 注意，上面的$\mathbf{X} \mathbf{w}$是一个向量，而$b$是一个标量。 

``` python
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
```



## 5. 定义损失函数`loss function`

因为需要计算损失函数的梯度，所以我们应该先定义损失函数。 这里我们使用平方损失函数。
$$
l^{(i)}(\mathbf{w}, b)=\frac{1}{2}\left(\hat{y}^{(i)}-y^{(i)}\right)^{2}
$$

n 个样本上的损失均值为：

$$
L(\mathbf{w}, b)=\frac{1}{n} \sum_{i=1}^{n} l^{(i)}(\mathbf{w}, b)=\frac{1}{n} \sum_{i=1}^{n} \frac{1}{2}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b-y^{(i)}\right)^{2}
$$

 在实现中，我们需要将真实值`y`的形状转换为和预测值`y_hat`的形状相同。

``` python
"""
    均方损失
"""
def squared_loss(y_hat, y):  # y_hat为估计量
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```



## 6. 定义优化算法（小批量随机梯度下降）

在每一步中，使用从数据集中随机抽取的一个小批量，然后根据参数计算损失的梯度。 接下来，朝着减少损失的方向更新我们的参数。 下面的函数实现小批量随机梯度下降更新。 该函数接受模型参数集合、学习速率和批量大小作为输入。每 一步更新的大小由学习速率`lr`决定。 因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（`batch_size`） 来规范化步长，这样步长大小就不会取决于我们对批量大小的选择。

``` python
"""
    小批量随机梯度下降
"""
def sgd(params, lr, batch_size):  # lr: 学习率，batch_size：批量大小
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # 梯度下降
            param.grad.zero_()
```



### 补充

#### `with`关键字用法

 `with`语句用于异常处理，封装了`try…except…finally`编码范式，提高了易用性。

我们拿常用的异常处理语句来类比

以下是`try...except...finally`语句：

``` python
file = open('./test_runoob.txt', 'w')
try:
    file.write('hello world')
finally:
    file.close()
```

以下是与之等价的`with`语句：

``` python
with open('./test_runoob.txt', 'w') as file:
    file.write('hello world !')
```

这两个语句是等价的。

[参考资料——python的with关键字](https://www.jianshu.com/p/5b01fb36fd4c)



#### `with torch.no_grad()`的使用

被`with torch.no_grad()`包住的代码，仅仅进行了计算，但不用跟踪反向梯度计算。

[参考资料——with torch.no_grad()的使用](https://zhuanlan.zhihu.com/p/386454263)



## 7. 训练

好了，前面的准备工作做完了，现在进入本篇文章的核心内容——训练

### 算法概括：

+ 初始化参数
+ 重复以下训练，直到完成
  + 计算梯度：$\mathbf{g} \leftarrow \partial_{(\mathbf{w}, b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l\left(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b\right)$
  + 更新参数：$(\mathbf{w}, b) \leftarrow(\mathbf{w}, b)-\eta \mathbf{g}$



> 上面式子的详细写法：
> $$
> \mathbf{w} \leftarrow \mathbf{w}-\frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b)=\mathbf{w}-\frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b-y^{(i)}\right),
> $$
>
> $$
> b \leftarrow b-\frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{b} l^{(i)}(\mathbf{w}, b)=b-\frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b-y^{(i)}\right)
> $$

在每个*迭代周期*（epoch）中，我们使用`data_iter`函数遍历整个数据集， 并将训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。 这里的迭代周期个数`num_epochs`和学习率`lr`都是超参数，分别设为3和0.03。 设置超参数很棘手，需要通过反复试验进行调整。 



训练结果：

```
epoch 1, loss 0.055445
epoch 2, loss 0.000249
epoch 3, loss 0.000050
```



## 8.完整代码（可运行）

``` python
import random
import torch
from d2l import torch as d2l
import numpy as np


"""
    根据带有噪声的线性模型构造一个人造数据集。
    我们使用线性模型参数w=[2,−3.4]⊤、b=4.2和噪声项ϵ生成数据集及其标签：
    y=Xw+b+ϵ
"""


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b  # 返回矩阵向量积
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))  # 返回一个列数为1的张量（行数由python解释器决定）


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
    线性回归模型
"""


def linreg(X, w, b):  # @save
    return torch.matmul(X, w) + b


"""
    均方损失
"""


def squared_loss(y_hat, y):  # y_hat为估计量
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


"""
    小批量随机梯度下降
"""


def sgd(params, lr, batch_size):  # lr: 学习率，batch_size：批量大小
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


"""
    生成数据集
"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)  # 生成一个含1000个数据的数据集

print("生成数据集test：")
print('features:', features[0], '\nlabel:', labels[0])  # Example
print()

# 散点图
d2l.set_figsize()  # 设置图片的大小
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()

"""
    读取数据集
"""
batch_size = 10
print('读取数据集test：')
# 注意循环最后一行的 break 实际上，这个循环只会运行一次，就退出循环了
for X, y in data_iter(batch_size, features, labels):
    print('X:\n', X, '\n', 'y:\n', y)
    break
print()

"""
    初始化模型参数
"""
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

"""
    训练
"""
lr = 0.03  # 学习率
num_epochs = 3  # 迭代周期个数
net = linreg
loss = squared_loss
print('训练test：')
# 进行迭代
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

