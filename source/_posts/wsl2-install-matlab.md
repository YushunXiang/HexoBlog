---
title: WSL2安装matlab R2020b
date: 2022-03-20 16:07:55
tags: 环境配置
categories: WSL2
---





软件与系统：Windows11下的WSL2(Windows Systemsystem for Linux)Ubuntu20.04和MATLAB R2020b

安装方法：将windows下载下载的MATLAB ISO挂载在新的磁盘分区，然后WSL的`/mnt`目录下安装MATLAB，并实现图形化显示。



## 下载MATLAB for Linux

百度网盘下载链接: https://pan.baidu.com/s/1g2USUdvb2DZ_2rktvSbCmw

密码: fs9k



下载完毕后，将以下四个文件解压（我这里用的解压软件是Bandizip）

![image-20220320163337415](image-20220320163337415.png)



解压完毕后会得到一个ISO文件：`Matlab99R2020b_Linux_64.iso`



## 挂载ISO到WSL下

我这里用`UltraISO`软件打开`.iso`文件，再添加虚拟光驱（我这里添加到了E盘）

![image-20220320163752749](image-20220320163752749.png)



将打开的ISO文件路径挂载到WSL系统上（WSL系统下的挂载目录为`/mnt/f`）

``` shell
sudo mkdir /mnt/f
sudo mount -t drvfs F:
```



如此，在WSL的`/mnt/f`目录下，就是MATLAB的虚拟光驱了。



## 安装MATLAB（此过程需要断网）

在WSL的`/mnt/f`目录下，安装MATLAB

``` shell
sudo ./install
```



填写安装密钥：`09806-07443-53955-64350-21751-41297`

选择文件许可证：在`MATLAB_R2020b_for_Linux补丁和证书.zip`的压缩包内有一个`license.lic`文件

后面就根据自己的需要选择软件安装路径和产品选择即可。



安装完成后取消挂载

``` shell
sudo umount /mnt/f
```



然后激活，把`MATLAB_R2020b_for_Linux补丁和证书`文件夹中的`libmwlmgrimpl.so`文件替换掉对应安装路径`bin/glnxa64/matlab_startup_plugins/lmgrimpl/`下的`libmwlmgrimpl.so`文件，可以用`sudo cp`指令



## 运行MATLAB

切换到安装路径的`/bin`文件夹下面，然后执行`sudo ./matlab`可运行MATLAB



成功示意图：

![image-20220320170459355](image-20220320170459355.png)



## 参考教程

[在Win 10子系统（WSL）中安装MATLAB](https://blog.csdn.net/budong_2017/article/details/112478811)

[ubuntu18.04安装matlab2020b](https://zhuanlan.zhihu.com/p/351907900)

