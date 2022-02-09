---
title: 多模态机器学习的调查和分类法
date: 2022-01-22 17:40:23
tags: 学习笔记
categories: 深度学习
---



[相关论文](https://eager-murdock-e61bfe.netlify.app/2022/01/22/multimodal-machine-learning-a-survey-and-taxonomy/Multimodal_Machine_Learning_A_Survey_and_Taxonomy.pdf)

## 1. Introduction

### A. Definition

A research problem or dataset is therefore characterized as *multimodal* when it includes multiple such modalities.

### B. Three modalities we focus primarily

1. natural language
2. visual signals
3. vocal signals

### C. Five challenges in Multimodal Machine Learning

也就是多模态研究的五个方向。

#### 1) Representation

Learning how to **represent and summarize multimodal data** in a way that exploits the complementarity（互补性） and redundancy（冗余性） of multiple modalities.

#### 2) Translation

**Translating(Mapping) data** from one modality to another.

#### 3) Alignment（对齐）

**Identifying the direct relations** between (sub)elements from two or more different modalities.

#### 4) Fusion （融合）

Joining information from two or more modalities to **perform a prediction**.

#### 5) Co-learning

**Transferring knowledge** between modalities, their representation, and their predictive models. 

![A summary of applications enabled by multimodal machine learning.](image-20220122191151114.png)

以上是对多模态研究的五个方向的简略介绍，将在下文详细介绍。



## 2. APPLICATIONS: A HISTORICAL PERSPECTIVE

Some examples:

1. audio-visual speech recognition (AVSR) 视听语音识别
2. the field of multimedia content indexing and retrieval 多媒体内容的索引和检索
3. visual question-answering 视觉问答



## 3. MULTIMODAL REPRESENTATIONS

多模态表示学习是指通过利用多模态之间的互补性，剔除模态间的冗余性，从而学习到更好的特征表示。



多模态表示分为两类：

### A. Joint representation 联合表示

概念：联合表示将多个模态的信息一起映射到一个统一的多模态向量空间，适用于推理期间存在所有模态的情况

应用： AVSR, affect, multimodal gesture recognition.

### B. Coordinated Representation 协同表示

概念：协同表示负责将多模态中的每个模态分别映射到各自的表示空间，但映射后的向量之间满足一定的相关性约束（例如线性相关）

应用：multimodal retrieval and translation,  grounding,  zero shot learning.

![A summary of multimodal representation techniques.](image-20220122223249302.png)



## 4. Translation (Mapping)

概念：转化也称为映射，负责将一个模态的信息转换为另一个模态的信息。

应用：主要应用于NLP。

早期研究：

+ speech synthesis
+ visual speech generation
+ video description
+ video description



多模态翻译可以分为两类：

### Exampled-based 基于实例的多模态翻译

![Example-based ](image-20220123172220582.png)

基于实例的多模态翻译从字典中检索最佳翻译，同时该算法受到训练数据字典的限制。

该算法有两种类型：

#### Retrieval-based models 基于检索的模型

它们依赖于在字典中找到最接近的样本，并将其用作翻译结果。检索可以在单模态空间或中间语义空间进行。

#### Combination-based models 基于组合的模型

基于组合的媒体描述方法的出发点是图像的句子描述具有共性和简单性可以利用的结构。



### Generative 生成式的多模态翻译

![Generative](image-20220123174008942.png)

生成式的多模态翻译首先在字典上训练翻译模型，然后使用该模型进行翻译。

关注点：

1. 语言
2. 视觉
3. 声音



该算法有三种方向：

#### Grammar-based models 基于语法的生成模型

依赖于预定义的语法来生成特定的模态。它们首先从源模式检测高级概念，例如图像中的对象和视频中的动作。然后将这些检测与基于预定义语法的生成过程结合在一起，生成目标模态。

优点：基于语法的生成模型使用预定义模板和受限制的语法时，它们更有可能生成语法上(对于语言)或逻辑上正确的目标实例。

缺点：基于语法的生成模型限制了他们产生公式化，而不是创造性翻译。此外，基于语法的方法依赖于复杂的==管道==进行概念检测，每个概念都需要单独的模型和单独的训练数据集。


#### Encoder-decoder models 编码器-解码器生成模型

该模型的主要思想是首先将源模态编码为矢量表示，然后使用解码器模块生成目标模态，所有这些都在一个单通道管道中。

#### Continuous generation models 连续生成模型



### 模型评价与讨论

难点：语音合成和媒体描述等任务没有一个标准的答案，很难对多模态翻译方法进行评价。故解决评价问题对多式翻译系统的进一步成功至关重要



## 5.  ALIGNMENT

从两个或多个模态中查找实例子组件之间的关系和对应

例子：

1. 给定一幅图像和一个标题，我们希望找到与标题的单词或短语对应的图像区域。
2. 给定一部电影，将其与剧本或书中它所基于的章节进行比对。

### Explicit alignment 显式对齐

来自两个或多个模式的实例子组件之间的对齐

#### 无监督

不使用直接对齐标签(即来自不同模式的实例之间的通信），解决了模式校准而无需任何直接校准标签。

#### （弱）监督

依赖于标记对齐的实例，用于训练用于对齐模式的相似性度量。

### Implicit alignment 隐式对齐

不显式地对齐数据，也不依赖于监督对齐示例，而是学习如何在模型培训期间对数据进行隐式对齐。

#### Graphical models 图形模型

用于更好地对齐机器翻译语言之间的单词和语音音素与其转录的对齐。需要人类手动定义。

#### Neural networks 神经网络

### 目前面临的困难

1. 具有显式标注对齐的数据集较少
2. 两种模式之间的相似度指标难以设计
3. 可能存在多种可能的对齐方式，一种模式中的元素不一定在另一种模式中都有对应关系



## 6. Fusion

将来自多种模态的信息集成在一起，并以预测结果为目标的概念:通过分类来预测一个类别(例如，快乐vs.悲伤)，或者通过回归来预测一个连续值(例如，情绪的积极性)。



### Model-agnostic approaches 模型不可知论方法

不直接依赖于特定机器学习方法的模型。



###  Model-based approaches 基于模型的方法

在构建中显式处理融合的基于模型的方法。更适用于多模态数据

#### Multiple kernel learning (MKL)  多核学习



#### Graphical models 图形模型



#### Neural Networks 神经网络



## 5. CO-LEARNING

 ![ Types of data parallelism used in co-learning](image-20220124002404774.png)

### Parallel data

并行模式来自相同的数据集，实例之间存在直接对应关系

#### Co-training 协同训练

是在多模态问题中只有少量的标记样本时，生成更多标记样本的过程

#### Transfer learning 转移训练

利用并行数据协同学习的另一种方法。



### Non-parallel data

分并行模式来自不同的数据集，没有重叠的实例，但在一般类别或概念上有重叠

#### Transfer learning 



#### Conceptual grounding 概念基础

指学习语义意义或概念，不仅仅是基于语言，还包括视觉、听觉、甚至是嗅觉等附加形式。



#### Zero shot learning (ZSL) 零距离学习

指在没有明确看到任何例子的情况下识别概念。



### Hybrid data

混合——实例或概念通过第三种模式或数据集进行桥接。



