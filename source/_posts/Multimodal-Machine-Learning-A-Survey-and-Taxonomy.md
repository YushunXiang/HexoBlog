---
title: 多模态机器学习的调查和分类法
date: 2022-01-22 17:40:23
tags: 学习笔记
categories: 强化学习
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



