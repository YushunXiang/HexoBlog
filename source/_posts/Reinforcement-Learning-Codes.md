---
title: 强化学习代码汇总
date: 2022-01-06 22:14:47
tags: 学习笔记
categories: 强化学习
math: true
---

不同的强化学习方法之间具有差异，除了言语表述外，用代码来理解不同学习方法的思路也是个好方法

## 基于 `SARSA` 算法的在线控制（同策学习）

关键公式：

$$
Q(S, A) \leftarrow Q(S, A)+ \alpha \left (R+ \gamma  Q \left(S^{\prime}, A^{\prime} \right)-Q(S, A) \right)
$$

![](image-20220106223847233.png)



## `SARSA(λ)` 算法流程

引入了资格迹

![](image-20220106224220249.png)



## 异策学习的 `Q-Learning` 的学习算法

关键公式：

$$
Q(S, A) \leftarrow Q(S, A)+ \alpha \left (R+ \gamma \max _{a} Q \left (S^{\prime}, a\right)-Q(S, A)\right)
$$

![](image-20220106224416585.png)