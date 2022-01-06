---
title: 强化学习学习笔记
date: 2022-01-06 21:21:00
tags: 学习笔记
categories: 强化学习
---

因为疫情的原因，本来在21年十二月末的强化学习考试推迟了。先准备结合老师上课PPT和一些教材以及网络视频，来总结一下强化学习的一些知识点与案例。



## 强化学习方法汇总



### 分类方法一：Model-Free 和 Model-Based

如果我们不尝试去理解环境, 环境给了我们什么就是什么. 我们就把这种方法叫做 model-free, 这里的 model 就是用模型来表示环境, 那理解了环境也就是学会了用一个模型来代表环境, 所以这种就是 model-based 方法. 



#### 基于`Model-Free`的方法

预测：

+ MC（蒙特卡洛学习）
+ TD（时序差分学习）



控制：

+ Q Learning
+ Sarsa
+ Policy Gradients



#### 基于`Model-Based`的方法

可以说，model-based的方法就属于Markov Decision Process（马尔可夫决策过程）



预测：

Dynamic Programming（基于模型的动态规划算法）



控制：

+ Policy Iteration
+ Value Iteration



### 分类方法二：基于概率和基于价值



#### 基于概率

根据概率采取行动, 所以每种动作都有可能被选中, 只是可能性不同。

如：Policy Gradients



#### 基于价值

基于价值的方法输出则是所有动作的价值, 我们会根据最高价值来选着动作。（感觉有点类似于贪婪策略？）

如：Sarsa, Q Learning



#### 一种结合了基于概率和基于价值的方法 `Actor-Critic`

actor 会基于概率做出动作, 而 critic 会对做出的动作给出动作的价值, 这样就在原有的 policy gradients 上加速了学习过程。



### 回合更新和单步更新

#### 回合更新

MC（蒙特卡洛学习）

Policy Gradients



#### 单步更新

TD（时序差分学习）

Q Learning

Sarsa



### 在线学习和离线学习

#### 在线学习

Sarsa



#### 离线学习

Q Learning



### 图表总结

| Sarsa      | Q Learning | Policy Gradients |
| ---------- | ---------- | ---------------- |
| Model-Free | Model-Free | Model-Free       |
| 基于价值   | 基于价值   | 基于概率         |
| 单步更新   | 单步更新   | 回合更新         |
| 在线学习   | 离线学习   |                  |

