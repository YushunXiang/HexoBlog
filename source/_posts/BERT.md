---
title: BERT
date: 2022-04-10 00:09:11
tags: 论文阅读
categories: 深度学习
---

[论文链接](https://arxiv.org/abs/1810.04805)

[TensorFlow代码链接](https://github.com/google-research/bert)

## BERT是什么

BERT的全称是**B**idirectional **E**ncoder **R**epresentations from **T**ransformers，是 Google 以无监督的方式利用大量**无标注**文本训练出来的语言模型，其架构为 Transformer 中的 Encoder（BERT=Encoder of Transformer）

（Google nb！）



## BERT的相对于ELMo、GPT的优点

![BERT vs ELMo and GPT](bert-1.png)

**ELMo**和**GPT**最大的问题就是传统的语言模型是单向的——我们根据之前的历史来预测当前词。但是我们不能利用后面的信息。传统的语言模型例如RNN是单向的，即只能利用单方向的信息。即使ELMo训练了双向的两个RNN，但是一个RNN只能看一个方向，因此也是无法 "同时" 利用前后两个方向的信息的。

但是基于**Transformer**架构的**BERT**，利用了**Self-Attention**机制，可以同时关注到前后的词。



## BERT的预处理（Pre-training BERT）

### Task 1: Masked Language Model

**随机遮盖或替换**一句话里面的任意字或词，然后让模型通过上下文预测那一个被遮盖或替换的部分，之后**做 Loss 的时候也只计算被遮盖部分的 Loss**，这其实是一个很容易理解的任务，实际操作如下：

1. 随机把一句话中 15% 的 token（字或词）替换成以下内容：
   1. 这些 token 有 80% 的几率被替换成 `[MASK]`，例如 my dog is hairy→my dog is [MASK]
   2. 有 10% 的几率被替换成任意一个其它的 token，例如 my dog is hairy→my dog is apple
   3. 有 10% 的几率原封不动，例如 my dog is hairy→my dog is hairy
2. 之后让模型**预测和还原**被遮盖掉或替换掉的部分，计算损失的时候，只计算在第 1 步里被**随机遮盖或替换**的部分，其余部分不做损失，其余部分无论输出什么东西，都无所谓



### Task2: Next Sentence Prediction

![BERT input representation](image-20220410003646656.png)

+ `[CLS]`表示句子的开头
+ `[SEP]`表示句子的结束
+ `[MASK]`表示利用`Masked Language Model`处理的词
+ `[PAD]`表示空白符



+ `Token Embedding`就是正常的词向量
+ `Segment Embedding`的作用是用embedding的信息让模型分开上下句
+ `Position Embedding`表示位置信息







## BERT的微调（Fine-tuning BERT）

讲真我不太理解

4种微调方式：

1. sentence pairs in paraphrasing
2. hypothesis-premise pairs in entailment
3. question-passage pairs in question answering
4. a degenerate text-∅ pair in text classifification or sequence tagging.