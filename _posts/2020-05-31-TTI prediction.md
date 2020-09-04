---
layout:     post
title:      "拥堵指数预测"
subtitle:   "深圳北站周边交通拥堵指数预测"
date:       2020-05-31
author:     "wumingyao"
header-img: "img/in-post/2020.05/31/bg.jpg"
tags: [交通拥堵预测,交通时空预测]
categories: [算法大赛]
---

## 主要内容
* [一、赛题说明](#p1)
* [二、数据说明](#p2)
* [三、Preliminary](#p3)
* [四、模型构建](#p4)
* [五、实验](#p5)
* [六、存在的问题](#p6)
* [七、挑战](#p7)
* [八、以往的工作进展(解决方案)](#p8)
* [九、引用](#p9)
## 正文

###  <span id="p1">一、赛题说明</span>
#### 【赛题背景】

深圳北站作为深圳市的大型交通枢纽，其周边交通运行的好坏直接决定到达深圳北站的通行效率；而深圳北站周边的福龙路、南坪快速路等快速路，又是市区内的主要运输通道，对深圳北站的出行造成了直接影响。
本赛题以此为背景，期望分析出深圳北站周边道路上，车辆对交通拥堵和北站出行的影响，实现基于网约车辆轨迹数据对路段交通拥堵指数的预测。

#### 【赛题任务】
本赛题任务是基于所给的数据对指定监测点在指定时刻的交通拥堵指数进行预测。

### <span id="p2">二、数据说明</span>
本赛题提供的训练集包括两个时间段，分别为从2019年1月到2019年3月，以及从2019年10月到2019年12月20日；空间范围限定在深圳北站周边，监测12个路段的交通拥堵指数。
训练集的数据由两类数据构成：交通拥堵指数数据，和网约车轨迹数据。 

1. 交通拥堵指数数据（train_TTI.csv）      
交通拥堵指数的计算时间粒度为10分钟，即每个拥堵指数的数值表示的是某路段连续10分钟的平均拥堵指数。此数据共有4个字段，分别是id_road、time、TTI和speed，字段说明如下
![Figure 1](../../../../../../img/in-post/2020.05/31/Figure 1.png)
数据样例如下：      
![Figure 2](../../../../../../img/in-post/2020.05/31/Figure 2.png)
路段范围和编号如下：      
![Figure 3](../../../../../../img/in-post/2020.05/31/Figure 3.png)

2. 网约车轨迹数据（train_201901_201903.zip, train_201910_201911.zip, train_20191201_20191220.zip）

此数据提供的是网约车的gps记录，这些gps记录按订单分组，每个订单都与上述路段相关。数据解压后为csv格式，有三个字段，分别为用户id、订单id以及轨迹记录，轨迹记录是一个复合字段，包含多组gps记录，每组gps记录包含有5个字段，分别是经度、纬度、速度、方向和时间戳，字段说明如下（字段名不包含在数据中，仅在此处说明）。

![Figure 4](../../../../../../img/in-post/2020.05/31/Figure 4.png)

数据样例如下：
![Figure 5](../../../../../../img/in-post/2020.05/31/Figure 5.png)


3. 测试集数据（toPredict_train_TTI.csv, toPredict_train_GPS.csv, toPredict_noLabel.csv）
预测2019/12/21~2020/01/01的指定时间片的TTI。

###  <span id="p3">三、Preliminary</span>
#### 3.1 Notations
假设有$N_g$条路段（本数据集有12条路段），每条路段的传感器产生$N_l$种时间序列（有两种时间序列，TTI和speed）。在这些序列中，指定一种时间序列做为预测的目标序列(TTI做为目标序列)，其他类型的序列做为特征。在给定长度为$T$的时间窗口的情况下，使用$Y=(y^1,y^2,\cdots,y^{N_g})\in \mathbb{R}^{N_g \times T}$来表示过去$T$个时间片内所有目标序列的读数（一共有$N_g$条路段），其中$y^i\in\mathbb{R}^T$属于路段$i$的目标序列读数。使用$X^i=(x^{i,1},x^{i,2},\cdots,x^{i,N_l})^T=(x_{1}^{i},x_{2}^{i},\cdots,x_{T}^{i})\in\mathbb{R}^{N_l\times T}$做为路段$i$的局部特征。其中，$x^{i,k}\in\mathbb{R}^T$是第$k$个被传感器记录的时间序列(本数据k=2,表示有两种时间序列，TTI和speed)，$x_{t}^{i}=(x_{t}^{i,1},x_{t}^{i,2},\cdots,x_{t}^{i,N_l})^T\in\mathbb{R}^{N_l}$代表$t$时刻路段$i$所有时间序列的读数。除了路段$i$的局部特征以外，由于不同路段之间的地理空间相关性，其他路段也共享大量对我们的预测有用的信息。最后，结合每个路段的局部信息到一个集合$\chi^i=\lbrace X^1,X^2,\cdots,X^{N_g} \rbrace$做为路段$i$的全局特征。

#### 3.2 Problem Statem
给定每条路段之前的读数和外部因素，预测路段$i$未来$\tau$个时间片的读数，标记为$\hat{\bf{y}}^i=(\hat{y}^i_{T+1},\hat{y}^i_{T+2},\cdots,\hat{y}^i_{T+\tau})\in\mathbb{R}^\tau$。

###  <span id="p4">四、模型构建</span>

#### 模型

![model](../../../../../../img/in-post/2020.05/31/model.jpg)
模型结构图

上图展示了本模型的结构图。本模型主要由三个重要部分所组成：

1） 多层注意力机制。该部分遵循编码器-解码器架构，我们使用两个独立的LSTM网络，一个用来编码输入的序列，例如，历史的TTI和speed时间序列，在编码器中，我们开发了两种不同的注意力机制，例如，局部空间注意力和全局空间注意力。另一个LSTM网络做为时间注意力解码器，用来预测序列$\hat{\bf{y}}^i$的输出序列。

2） 时空残差网络。由于交通条件的变化会受到基础道路网的限制，所以将路网信息纳入预测模型至关重要。所以我们将静态的道路网进行空间嵌入。同时，我们使用one-hot编码将每个时间步的星期几和时间编码为$\mathbb{R}^7$和$\mathbb{R}^T$，并将他们连接到向量$\mathbb{R}^{T+7}$做为时间嵌入。并将空间嵌入和时间嵌入连接全局特征，输入到ResNet中进行训练。

3） 外部因素融合。此模块用于处理外部因素的影响，其输出作为其输入的一部分喂给解码器。

### <span id="p5">五、实验</span>
![experience](../../../../../../img/in-post/2020.05/31/experience.jpg)

###  <span id="p6">六、存在的问题</span>

#### 盲目搭建模型
目前我们搭建的模型其实只是将几个模型杂糅到一块，例如ST-RestNet是来自ADST模型的，多层注意力机制是来自GeoMan模型的，
编码器-解码器是来自seq2seq模型的，这几个部分对我们的预测是不是有用的，我们也只是通过实验结果进行验证。
这几个部分为什么有用，以及这几个部分分别解决了我们大问题中的哪些小问题（例如动态时空相关性，累计误差等），以及目前解决
这些小问题最好的模型是不是这些，我们之前并没有考虑。我们只是盲目性地将我们学到的一些模型搭在了一起。

我们需要将我们所研究的问题分解成几个子问题，然后通过一些顶级期刊或者顶级会议的模型对应解决这几个子问题，
并且从这里面挖掘出我们的创新点。

#### 数据集的问题
我们目前只有一份交通拥堵相关的数据集，至少还需要另外一份数据集，如果没有交通拥堵的数据，可以考虑使用交通流的数据集。

#### 需要补充的基线实验
(1)Arima     
(2)SVR       
(3)前馈神经网络FNN       
(4)FC-LSTM 它是一个在编码器和解码器中具有完全连接的LSTM层的序列到序列模型          
(5)STGCN         
(6)DCRNN扩散卷积递归神经网络           
(7)WaveNet        
(8)GMAN          
(9)GeoMan

#### 评价指标
MAE、RMSE、MAPE

### <span id="p7">七、挑战</span>
#### 7.1 复杂的时空相关性。[1,2,4]
![Figure 1](../../../../../../img/in-post/2020.05/16/Figure 1.jpg)
Figure1:Complexspatio-temporalcorrelations.

1) 动态的空间相关性。

如图1所示，相邻区域的交通数据具有一定的相关性。
但随着时间的推移，路网中传感器之间的交通条件相关性发生了重大变化。
可以这么理解：每个时段一个路口左拐跟右拐的比例肯定是动态变化的，从而导致空间相关性上的动态变化。图1的Spatial correlation 也就是那些实线的颜色实际上是随着时间变化而变化的。
所以，如何动态选择相关传感器的数据来预测目标传感器在长期范围内的交通状况是一个具有挑战性的问题。

2) 非线性时间相关性。

不同的时间间隔里交通状态之间也存在着相关性。
例如：t-1时刻传感器3处的交通状况由于交通事故发生堵塞，那么，t-1时刻对后面的影响可能会比较大。
所以，传感器3在时间步t+l+1的交通状况可能与较远的时间步(如t-1)的交通状况相关，而与较近的时间步长(如t+l)的交通状况不大相关。
如何随时间的推移下，对非线性时间相关性进行自适应建模仍是一个挑战。

#### 7.2 如何处理累计误差传播问题?[4]

传统的时间预测的方法采用的是逐步预测的方法，先预测下一步，再用下一步预测下下一步。因此，逐步预测会产生累计误差。
#### 7.3 历史交通数据具有周期性，如何提取周期性特征。[3]
交通数据通常显示出明显的周期性模式。在一天的24小时内，通常会有一两个高峰时段发生交通状况拥堵。每周的交通数据也会显示出周期性变化，例如工作日的交通数据会不同于周末。

#### 7.4 外界因素
天气，节假日……

### <span id="p8">八、以往的工作进展(解决方案)<span>
#### 8.1 复杂的时空相关性。
Zheng[4]提出的GMAN模型，构建了时空注意机制来模拟复杂的时空相关性，

### <span id="p9">九、引用</span>
[1] J. Wang, Q. Gu, J. Wu, G. Liu and Z. Xiong, "Traffic Speed Prediction and Congestion Source Exploration: A Deep Learning Method," 2016 IEEE 16th International Conference on Data Mining (ICDM), Barcelona, 2016, pp. 499-508, doi: 10.1109/ICDM.2016.0061.

[2] Lv Z, Xu J, Zheng K, et al. Lc-rnn: A deep learning model for traffic speed prediction[C]//IJCAI. 2018: 3470-3476.

[3] J. Tang, F. Liu, Y. Zou, W. Zhang and Y. Wang, "An Improved Fuzzy Neural Network for Traffic Speed Prediction Considering Periodic Characteristic," in IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 9, pp. 2340-2350, Sept. 2017, doi: 10.1109/TITS.2016.2643005.

[4] Zheng C, Fan X, Wang C, et al. Gman: A graph multi-attention network for traffic prediction[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(01): 1234-1241.

[5] M. Cao, V. O. K. Li and V. W. S. Chan, "A CNN-LSTM Model for Traffic Speed Prediction," 2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring), Antwerp, Belgium, 2020, pp. 1-5, doi: 10.1109/VTC2020-Spring48590.2020.9129440.
