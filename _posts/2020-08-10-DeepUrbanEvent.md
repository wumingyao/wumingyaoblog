---
layout:     post
title:      "DeepUrbanEvent"
subtitle:   "A System for Predicting Citywide Crowd Dynamics at Big Events"
date:       2020-08-10
author:     "wumingyao"
header-img: "img/in-post/2020.08/10/bg.jpg"
tags: [论文笔记,]
categories: [论文笔记]
---

## 主要内容
* [ABSTRACT](#p1)
* [一、INTRODUCTION](#p2)
* [二、CITYWIDE CROWD DYNAMICS MODELING](#p3)
* [三、DEEP SEQUENTIAL LEARNING ARCHITECTURE](#p4)
* [四、EXPERIMENT](#p5)

## 参考论文：
[《DeepUrbanEvent:A System for Predicting Citywide Crowd Dynamics at Big Events》](https://www.researchgate.net/profile/Renhe_Jiang/publication/334714928_DeepUrbanEvent_A_System_for_Predicting_Citywide_Crowd_Dynamics_at_Big_Events/links/5d417167299bf1995b597f28/DeepUrbanEvent-A-System-for-Predicting-Citywide-Crowd-Dynamics-at-Big-Events.pdf)

## 正文

###  <span id="p1">ABSTRACT</span>
#### 任务(Objective)
任务：当地震、台风或者传统节日等事件发生时,对短期内的人流趋势进行预测。
人群趋势分为两部分：人群密度和人群流。
#### 方法(Methods)：对研究的基本设计加以描述。
作者搭建了一个在线系统DeepUrbanEvent,可以迭代地将当前一小时的全市人群动态作为输入，并将下一小时的预测结果作为输出。
设计了一种新的基于递归神经网络的深度学习体系结构，以类似于视频预测任务的方式对这些高度复杂的序列数据进行有效建模。
#### 结果(Results)
实验结果证明作者提出的方法比现有的方法表现更好。
#### 结论(Conclusion)
作者将他们的原型系统应用到多个大型的真实事件中，
并证明它作为一个在线人群管理系统具有很高的可部署性。
###  <span id="p2">一、INTRODUCTION</span>
#### 研究背景和重要性(Background And Importance)
背景：活动人流管理已经成为一个具有高度社会影响的研究课题了。
当一些大事件发生时，

1)当地震、台风或者传统节日来临时，对于政府和公共服务管理者来说，保护人们安全和
维护公共基础设施正常运行时首要任务。尤其对于有些人口密集的大城市来说。

2)AI技术的发展和5G的来临，大量的人类移动数据通过各种源头持续产生，这些数据可以被
用来做为流数据来预测人群的动态变化。
#### 该领域的科研空白/挑战(Description of knowledge gap or chanllenge)
当大的事件或者灾难发生时，人类的移动可能会发生急剧的变化，这意味着人们的移动和平时的
路线就不相干了。所以在事件发生的情况下预测人群的动态变化是非常具有挑战性的。
#### 本文的研究课题(Topic Of Research Paper)
在城市范围内，当事件发生的情况下预测人群的动态变化。

只从当前观测中提取深层次的人流趋势和产生一个准确的短期人流趋势预测结果。
搭建了一个基于所收集的大量的人流数据和一个深度学习框架的在线系统DeepUrbanEvent，
可以迭代地将当前一小时的全市人群动态作为输入，并将下一小时的预测结果作为输出。

详细描述

1)全市人群动态被分为两部分：人群密度和人群流。

2)将城市划分为细粒度的网格，并使用类似于视频的一个四维的张量(Timestep,Height,Width,Channel)表示，
其中Timestep代表观测或者预测的步数，Height,Width是匹配大小，对于每个匹配网格，人群密度/人群流量视频代表
一个密度/流量值的时间序列，因此对于密度的Channel为1，而对于流量的Channel等于流量
核窗口$\eta\times\eta$的大小。所存储的值表示在所给的时间间隔内，中心网格内有多少人将转移到
$\eta\times\eta$相邻网格中的每个网格。
#### 核心方法论和主要发现/结果(Highlight The Approach And Highlight Principal Findings/Result)
核心方法：设计了一个多任务ConvLSTM Encoder-Decoder架构来同时建模这两种高维度的序列
数据来获得并发增强。

结果：使用了四个真实的发生在东京的事件数据集，即3.11 JapanEarthquake,Typhoon Roke(2011),New Year’s Day(2012),and Tokyo Marathon(2011),
进行验证，并且证明了其优于现有基线。

主要发现：

1) 对于预测在大型的事件发生时预测人流的动态变化，作者搭建了一个只需要有限的当前观测做为输入的在线部署的系统

2) 城市人群动态变化被分解成两种人造视频，即人群密度视频核人群流量视频。一个多任务的
ConvLSTM Encoder-Decoder 架构网络被设计来同时预测这两种变化。

3) 通过大型的真实数据集验证了该模型，并且部署到了原型系统上。


###  <span id="p3">二、CITYWIDE CROWD DYNAMICS MODELING</span>

#### 定义
1) Calibrated human trajectory dataset:在数据集$\tau$中，人的轨迹
通过day(i)核userid(u)存储并标定。给定一个区域的网格M,即一个网格集合
$\lbrace g_1,g_2,\cdots,g_m,\cdots,g_{Height*Width},\rbrace$,
一个时间间隔$\Delta t$。每个用户在一天的轨迹$\tau_{iu}$:

$$\tau_{iu}=(t_1,g_1),\cdots,(t_k,g_k) \bigwedge \forall j\in[1,k),|t_{j+1}-t_j|=\Delta t$$

以下只考虑轨迹数据库$\tau$的一天切片，那么当参考$\tau$时，可以省略日索引(i)。

2) Crowd density:在给定$\tau,M$,人群密度在时间片$t$网格$g_m$上定义为：

$$d_{tm}=|\lbrace u|\tau_u\cdot g_t=g_m\rbrace|$$

表示在$t$时刻$g_m$里面有多少人。

3) Crowd flow:为了计算从一个中心网格开始的人群流量，作者使用一个$\eta\times\eta$的核窗口，
它表示由$\eta\times\eta$相邻网格组成的正方形区域，$g_m$是质心。

$$f_{tmw}=|\lbrace u|\tau_u\cdot g_{t-1}=g_m\bigwedge\tau_u\cdot g_t=g_w\rbrace||$$

表示在$t-1$时刻网格$g_m$有多少人转移到了邻居网格$g_w$。

4) Crowd density/flow video:因为网格被标示为2维格式，crowd density/flow video 包含
一组连续的帧可以被表示为4维张量$\mathbb{R}^{Timestep\times Height\times Width\times Channel}$,
其中，Timestep表示视频帧的数量，density的Channel为1，flow的Channel等于被给的核
窗口的大小，即$\eta^2$。Figure 1描述了Crowd density/flow video。
![Figure 1](../../../../../../img/in-post/2020.08/10/Figure 1.jpg)

5) Crowd density/flow video prediction:在给定当前a-step的crowd density/flow video观测
$x_d=d_{t-(\alpha-1),\cdots,d_t},x_f=f_{t-(\alpha-1),\cdots,f_t}$,
对于下一个$\beta$-step density/flow video $\hat y_d=\hat d_{t+1},\cdots,\hat d_{t+\beta}$,
$\hat y_f=\hat f_{t+1},\cdots,\hat f_{t+\beta}$。建模如下：

![math](../../../../../../img/in-post/2020.08/10/math.jpg)

6) Citywide crowd dynamics prediction:同时预测density/flow

$$\hat y_d,\hat y_f=\mathop{argmax}\limits_{y_d,y_f} P(y_d,y_f|x_d,x_f)$$
![Figure 3](../../../../../../img/in-post/2020.08/10/Figure 3.jpg)
Figure3:Citywide Crowd Dynamics Prediction.

###  <span id="p4">三、DEEP SEQUENTIAL LEARNING ARCHITECTURE</span>

Convolutional LSTM:ConvLSTM被提出来建立一个从端到端的可训练的模型，用于降雨预报。
它扩展了一个全连接的LSTM(FC-LSTM)在输入到状态和状态到状态的转换中都有卷积结构，并实现了新的成功视频建模任务。
ConvLSTM有三个门机制，包含一个输入门$i$,一个输出门$o$,和一个遗忘门$f$。
对于一个输入帧序列$(x_1,x_2,\cdots,x_T)$来说，在ConvLSTM中的隐藏状态$h_t$被从t=1到T迭代计算

$$i_t=\sigma(W_{x_i}*x_t+W_{h_i}*h_{t-1}+W_{c_i}\odot c_{t-1}+b_i)$$

$$f_t=\sigma(W_{x_f}*x_t+W_{h_f}*h_{t-1}+W_{c_f}\odot c_{t-1}+b_f)$$

$$c_t=f_t\odot c_{t-1}+i_t\odot tanh(W_{x_c}*x_t+W_{h_c}*h_{t-1}+b_c)$$

$$o_t=\sigma(W_{x_o}*x_t+W_{h_o}*h_{t-1}+W_{c_o}\odot c_t+b_o)$$

$$h_t=o_t\odot tanh(c_t)$$

其中，$W$是权重，$b$是偏置矩阵，*代表卷积操作，$\odot$是哈达玛乘积。
![Figure 4](../../../../../../img/in-post/2020.08/10/Figure 4.jpg)
Figure 4: Stacked ConvLSTM for One-Step Prediction.

#### 3.1 Stacked ConvLSTM Architecture
![3.1](../../../../../../img/in-post/2020.08/10/3.1.jpg)

#### 3.2 CNN AutoEncoder for Crowd Flow
flow的Channel比density的要大很多。在该系统中，每个网格是500m$\times$500m,
考虑到可能的交通方式如步行、公共汽车、汽车和火车，在5分钟的时间间隔内，
从一个网格单元到另一个网格单元的转换距离可以达到4km。所以设置核窗口为15$\times$15
尽可能的捕捉5分钟内的人群流量。所以flow的channel等于255，对于目前最优的视频学习模型
来说太大了，所以需要搭建一个CNN AutoEncoder来获取低维度的表达。

![Figure 5](../../../../../../img/in-post/2020.08/10/Figure 5.jpg)
Figure5:CNN Auto Encoder for CrowdFlow.

对于CNN层来说，在(i,j)位置的第k-th层的卷积特征值为：

$$c_{i,j,k}=ReLU(w_kx_{i,j}+b_k)$$

其中，$x_{i,j}$是位置(i,j)的输入。

the special CNN AutoEncoder的细节如Fig.5。原始人群流图像用三维张量（15,15,225）表示，
一个由3个卷积层构成的增量将图像表示为全部的三维张量（15,15,4），
然后用3个卷积层构造解码器将压缩后的传感器解码到原始三维张量（15,15,225）。
端到端模型参数可以通过最小化原始流图像和之间的重建误差（MSE）来优化解码流图像，
我们可以得到压缩通道（从225到4），但保持流图像的空间结构信息在（15,15），
因此，只需要用1x1的卷积核窗口。在编码器的最后一层,使用了一个独特的ReLU（MAX=1.0）函数，以确保值完全可伸缩到[0,1]，这可以帮助众流的值范围近似等于密度的值范围。

#### 3.3 Multitask ConvLSTM Encoder and Decoder
因为多个单步预测会造成累计误差，所以使用encoder-decoder架构。

通过最小化预测误差$L(\theta_d)$和$L(\theta_f)$，可以分别训练ConvlTM Enc.-Dec.人群密度和流量模型，描述如下：

$$L(\theta_d)=||\hat{\mathbb{Y}}_D-\mathbb{Y}_D||^2$$

$$L(\theta_f)=||\hat{\mathbb{Y}}_F-\mathbb{Y}_F||^2$$

人群密度视频和人流视频共享重要信息和高分彼此都有关系。他们的想法后面有两层:
(1)人流趋于顺势，特别是在紧急情况下，这可能会使拥挤的地方吸引越来越多的人；
(2)较高的入流将导致网格单元的高密度，而较高的流出将降低人群密度。
![Figure 6](../../../../../../img/in-post/2020.08/10/Figure 6.jpg)

整个模型通过最小化预测误差$L(\theta)$来对模型进行训练

$$L(\theta)=\lambda_d||\hat{\mathbb{Y}}_D-\mathbb{Y}_D||^2+\lambda_f||\hat{\mathbb{Y}}_F-\mathbb{Y}_F||^2$$

其中$\lambda_d和\lambda_f都等于0.5$

###  <span id="p5">四、EXPERIMENT</span>
#### 实验建立
选择了东京做为目标城市。选择了该地区四个城市级别的事件做为测试：
(1)3.11 Earthquake(2011/03/11),
(2)Typhoon Roke(2011/09/21),
(3)NewYear’s Day (2012/01/01),
(4)Tokyo Marathon(2011/02/27).

活动日前连续10天作为培训和验证数据集，其中包括：
2011/03/01-2011/03/10,
2011/09/11-2011/09/20,
2011/12/222011/12/31和
2011/02/17-2011/02/26。

源数据包括将近100000~130000个用户每一天的GPS记录。在进行数据清洗和噪声处理后，对每个
用户的GPU记录进行每5min采样。

#### 参数设置
每个格子大小为$\Delta Long$=0.005，$\Delta Lat$=0.004(500mx500m)，
一共产生80x80的格子地图。时间间隔$\Delta t$设置为5min。因此有2880个时间片(288*10days)
做为训练数据集，288个时间片做为测试数据集。crowd flow的窗口设置为15x15,
设置$\alpha,\beta$分别为6.
最后一共有2868个样本。80%用于训练，20%用于验证。
其中，batch_size=4,lr=0.0001。CNNAutoEncoder的lr=0.001。
epochs_max=200。
对于crowd density，使用500做为缩小因子，
对于crowd flow，使用100做为缩小因子。
