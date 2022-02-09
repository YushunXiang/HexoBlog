---
title: PyTorch-softmax回归的从零开始实现
date: 2022-01-27 21:03:04
tags: 学习笔记
categories: 深度学习
math: true
---



回归可以用于预测*多少*的问题。 比如预测房屋被售出价格，或者棒球队可能获得的胜场数，又或者患者住院的天数。

事实上，我们也对*分类*问题感兴趣：不是问“多少”，而是问“哪一个”：

- 某个电子邮件是否属于垃圾邮件文件夹？
- 某个用户可能*注册*或*不注册*订阅服务？
- 某个图像描绘的是驴、狗、猫、还是鸡？
- 某人接下来最有可能看哪部电影？

通常，机器学习实践者用*分类*这个词来描述两个有微妙差别的问题： 1. 我们只对样本的“硬性”类别感兴趣，即属于哪个类别； 2. 我们希望得到“软性”类别，即得到属于每个类别的概率。 这两者的界限往往很模糊。其中的一个原因是：即使我们只关心硬类别，我们仍然使用软类别的模型。

所以我们引入了softmax回归。



## 0. 准备

导包：

``` python
import torch
from IPython import display
from d2l import torch as d2l
```

引入`Fashion-MINIST`数据集：

``` python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

关于`d2l`包中内置的`load_data_fashion_mnist`函数：

``` python
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```





## 1. 初始化模型参数

这里的每个样本都将用固定长度的向量表示。 原始数据集中的每个样本都是$28 \times 28$的图像。 在本节中，我们将展平每个图像，把它们看作长度为784的向量。 我们暂时只把每个像素位置看作一个特征。



回想一下，在`softmax`回归中，我们的输出与类别一样多。 因为我们的数据集有10个类别，所以网络输出维度为10。 因此，权重将构成一个$784 \times 10$的矩阵， 偏置将构成一个$1 \times 10$的行向量。 与线性回归一样，我们将使用正态分布初始化我们的权重`W`，偏置初始化为0。

``` python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```



## 2. 定义softmax操作

softmax表达式：
$$
\operatorname{softmax}(\mathbf{X})_{i j}=\frac{\exp \left(\mathbf{X}_{i j}\right)}{\sum_{k} \exp \left(\mathbf{X}_{i k}\right)}
$$




实现softmax由三个步骤组成：

1. 对每个项求幂（使用exp）
2. 对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数
3. 将每一行除以其规范化常数，确保结果的和为1



``` python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
```



## 3. 定义模型

softmax回归的矢量计算表达式为：
$$
\begin{array}{l}
\mathbf{O}=\mathbf{X W}+\mathbf{b} \\
\hat{\mathbf{Y}}=\operatorname{softmax}(\mathbf{O})
\end{array}
$$


定义softmax操作后，我们可以实现softmax回归模型。 下面的代码定义了输入如何通过网络映射到输出。 注意，将数据传递到模型之前，我们使用`reshape`函数将每张原始图像展平为向量。

``` python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```



## 4. 定义损失函数

我们引入交叉熵损失函数

softmax函数给出了一个向量$\hat{\mathbf{y}}$， 我们可以将其视为“对给定任意输入$\mathbf{x}$的每个类的条件概率”。 例如$\hat{y}_{1}=P(y=\text { 猫 } \mid \mathbf{x})$。 假设整个数据集${\mathbf{X},\mathbf{Y}}$具有$n$个样本， 其中索引ii的样本由特征向量$\mathbf{x}^{(i)}$和独热标签向量$\mathbf{y}^{(i)}$组成。 我们可以将估计值与实际值进行比较：
$$
P(\mathbf{Y} \mid \mathbf{X})=\prod_{i=1}^{n} P\left(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}\right)
$$
根据最大似然估计，我们最大化$P(\mathbf{Y} | \mathbf{X})$，相当于最小化负对数似然：
$$
-\log P(\mathbf{Y} \mid \mathbf{X})=\sum_{i=1}^{n}-\log P\left(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}\right)=\sum_{i=1}^{n} l\left(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}\right)
$$
其中，对于任何标签$\mathbf{y}$和模型预测$\hat{\mathbf{y}}$，损失函数为：
$$
l(\mathbf{y}, \hat{\mathbf{y}})=-\sum_{j=1}^{q} y_{j} \log \hat{y}_{j}
$$

``` python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
```



## 5. 分类精度

当预测与标签分类`y`一致时，即是正确的。 分类精度即正确预测数量与总预测数量之比。 虽然直接优化精度可能很困难（因为精度的计算不可导）， 但精度通常是我们最关心的性能衡量标准，我们在训练分类器时几乎总会关注它。

为了计算精度，我们执行以下操作。 首先，如果`y_hat`是矩阵，那么假定第二个维度存储每个类的预测分数。 我们使用`argmax`获得每行中最大元素的索引来获得预测类别。 然后我们将预测类别与真实`y`元素进行比较。 由于等式运算符“`==`”对数据类型很敏感， 因此我们将`y_hat`的数据类型转换为与`y`的数据类型一致。 结果是一个包含0（错）和1（对）的张量。 最后，我们求和会得到正确预测的数量。

``` python
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)
```



同样，对于任意数据迭代器`data_iter`可访问的数据集， 我们可以评估在任意模型`net`的精度。

```python
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```







## 参考

[参考教程：DIVE INTO DEEP LEARNING](https://zh-v2.d2l.ai/chapter_linear-networks/linear-regression-scratch.html)

