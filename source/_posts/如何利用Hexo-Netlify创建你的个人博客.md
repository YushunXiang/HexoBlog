---
title: 如何利用Hexo + Netlify创建你的个人博客
date: 2022-01-02 18:38:46
tags:
---

因为疫情封宿舍了，闲来无聊决定创建一个网站，来随便记录一下日常生活。



## 在创建博客之前的准备工作

安装以下两个软件

[Node.js](https://nodejs.org/en/)

[Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)



## 创建一个Hexo博客

### 一、启动Hexo博客



打开`powershell`，安装`hexo-cli`

```
 npm install hexo-cli
```



在合适的位置创建一个新文件夹，用于存放Hexo博客文件

``` powershell
mkdir <folderName>
cd <folderName>
```

我这里的`<folderName>`为`HexoBlog`，各位也可以自己取名字。



在刚刚的克隆下来的仓库的根目录，输入

``` powershell
hexo init
```

此步完成后，你将看到如下内容

![image-20220102175024199](%E5%88%9B%E5%BB%BA%E5%8D%9A%E5%AE%A2.assets/image-20220102175024199.png)



接下来，运行此命令以完成安装依赖项

``` powershell
npm install
```

此步完成后，你将看到如下内容

![image-20220102175309803](%E5%88%9B%E5%BB%BA%E5%8D%9A%E5%AE%A2.assets/image-20220102175309803.png)



访问你的本地Hexo Blog

``` powershell
 hexo server
```

![image-20220102175430091](%E5%88%9B%E5%BB%BA%E5%8D%9A%E5%AE%A2.assets/image-20220102175430091.png)

可以通过 `Ctrl` + 鼠标右键 访问



### 二、创建GitHub仓库

![image-20220102172443485](%E5%88%9B%E5%BB%BA%E5%8D%9A%E5%AE%A2.assets/image-20220102172443485.png)



取一个合适的名字，我这里为`HexoBlog`



### 三、添加远程库

选择一个合适的目录

在`powershell`中依次输入

``` powershell
git init
git add .
git commit -m "initial commit"
git remote add origin <your SSH code>
git push origin master
```



其中，`<your SSH code>`为下图所示的内容，我这里是`git@github.com:YushunXiang/HexoBlog.git`

![image-20220102172957834](%E5%88%9B%E5%BB%BA%E5%8D%9A%E5%AE%A2.assets/image-20220102172957834.png)

这样，你就把本地的代码库与GitHub上的远程库关联在一起了



### 四、设置Netlify

在[Netlify官网](https://www.netlify.com/)创建一个账户



授权 Netlify 访问您的 Github 存储库，并进行设置

![image-20220102181627363](%E5%88%9B%E5%BB%BA%E5%8D%9A%E5%AE%A2.assets/image-20220102181627363.png)



稍等片刻，你的网站就构建好了！

下图的框出部分就是你的网站网址了！你可以点击访问。

![image-20220102182225569](%E5%88%9B%E5%BB%BA%E5%8D%9A%E5%AE%A2.assets/image-20220102182225569.png)