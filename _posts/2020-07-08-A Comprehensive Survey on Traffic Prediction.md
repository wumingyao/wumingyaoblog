---
layout:     post
title:      "交通预测综述"
subtitle:   "A Comprehensive Survey on Traffic Prediction"
date:       2020-07-08
author:     "wumingyao"
header-img: "img/in-post/2020.07/08/bg.jpg"
tags: [交通预测,论文笔记,综述]
categories: [论文笔记]
---

## 参考论文：
[《A Comprehensive Survey on Trafﬁc Prediction》](https://arxiv.org/pdf/2004.08555)


## 主要内容
* [一、Abstract](#p1)
* [二、INTRODUCTION](#p2)
* [三、TRADITIONAL METHODS](#p3)
* [四、DEEP LEARNING METHODS](#p4)
* [五、THE STATE-OF-THE-ART RESULTS](#p5)
* [六、PUBLIC DATASETS](#p6)
* [七、 FUTURE DIRECTION AND OPEN PROBLEMS](#p7)
* [八、REFERENCES](#p8)
## 正文

###  <span id="p1">一、Abstract</span>
交通预测在智能交通系统中起着至关重要的作用。准确的交通预测可以辅助路线规划，指导车辆调度，缓解交通拥堵。
由于路网中不同区域之间复杂而动态的时空依赖性，这一问题具有挑战性。
近年来，大量的研究工作投入到这一领域，极大地提高了交通预测能力。
本文的目的是为交通预测提供一个全面的调查。具体来说，本文首先总结了现有的流量预测方法，并对其进行了分类。
其次，本文列举了交通量预测的一般任务以及这些任务的最新研究现状。
第三，本文收集和整理现有文献中广泛使用的公共数据集。
此外，本文通过大量的实验对两个数据集上的交通需求和速度预测方法的性能进行了比较。
最后，本文讨论了未来可能的发展方向。

###  <span id="p2">二、INTRODUCTION</span>
现代城市正在逐步发展成为智慧城市。城市化进程的加快和城市人口的快速增长给城市交通管理带来了巨大的压力。
智能交通系统（ITS）是智能城市不可缺少的组成部分，交通预测是ITS发展的基石。
准确的交通量预测对于许多实际应用是必不可少的。例如，交通量预测有助于缓解城市交通拥堵；
租车需求预测可以促使汽车共享公司将汽车预先分配到高需求地区。日益增长的可用交通相关数据集为探索这一问题提供了潜在的新视角。

#### 2.1 主要任务
现有交通量预测工作的主要应用任务如下：

##### 2.1.1 流量预测(Flow)

交通流量预测就是预测未来一段时间内通过道路上某一点的车辆数量。

##### 2.1.2 速度预测(Speed)

交通速度预测就是预测未来一段时间内道路上的车辆平均车速。

##### 2.1.3 需求量预测(Demand)

交通需求包括出租车和共享单车的需求。交通需求量预测就是使用历史数据来预测一个区域在未来一段时间中的需求量。

##### 2.1.4 占用率预测(Occupancy)

占用率预测指的是预测未来一段时间内车辆占用道路空间的程度。在测量时，它还考虑了交通组成和速度的变化，并提供了更可靠的车辆占用道路的程度指标。

##### 2.1.5 旅行时间预测(Travel time)

在获取路网中任意两点的路线的情况下，预测从路线中的一个点到另一个点的旅行时间。


#### 2.2 挑战
交通预测具有很大的挑战性，主要受以下复杂因素的影响：

![Figure 1](../../../../../../img/in-post/2020.05/16/Figure 1.jpg)
Figure1:Complexspatio-temporalcorrelations.

##### 2.2.1 动态的空间相关性。
如图1所示，相邻区域的交通数据具有一定的相关性。
但随着时间的推移，路网中传感器之间的交通条件相关性发生了重大变化。
可以这么理解：每个时段一个路口左拐跟右拐的比例肯定是动态变化的，从而导致空间相关性上的动态变化。图1的Spatial correlation 也就是那些实线的颜色实际上是随着时间变化而变化的。
所以，如何动态选择相关传感器的数据来预测目标传感器在长期范围内的交通状况是一个具有挑战性的问题。

##### 2.2.2 非线性时间相关性。
不同的时间间隔里交通状态之间也存在着相关性。
例如：t-1时刻传感器3处的交通状况由于交通事故发生堵塞，那么，t-1时刻对后面的影响可能会比较大。
所以，传感器3在时间步t+l+1的交通状况可能与较远的时间步(如t-1)的交通状况相关，而与较近的时间步长(如t+l)的交通状况不大相关。
如何随时间的推移下，对非线性时间相关性进行自适应建模仍是一个挑战。

##### 2.2.3 历史交通数据具有周期性，如何提取周期性特征。
交通数据通常显示出明显的周期性模式。在一天的24小时内，通常会有一两个高峰时段发生交通状况拥堵。每周的交通数据也会显示出周期性变化，例如工作日的交通数据会不同于周末。

##### 2.2.4 外部因素
交通时空序列数据还受到一些外部因素的影响，如天气状况、交通事件或道路属性等。

##### 2.2.5 如何处理累计误差传播问题?
传统的时间预测的方法采用的是逐步预测的方法，先预测下一步，再用下一步预测下下一步。因此，逐步预测会产生累计误差。

综上所述，交通数据在空间和时间两个维度上都表现出很强的动态相关性。
因此，如何挖掘非线性、复杂的时空模式，从而进行准确的交通预测是一个重要的课题。

#### 2.3 交通预测相关调查
最近有一些调查从不同的角度回顾了特定环境下交通预测的文献。
[2] 回顾了2004年至2013年的方法和应用，并讨论了当时十分重要的十大挑战。
目前的研究主要集中在考虑短期交通量预测方面，相关文献主要基于传统方法。
另一项工作[3]也关注短期交通量预测，简要介绍了交通量预测所采用的技术，并提出了一些研究建议。
[4] 概述了交通量预测的意义和研究方向。
[5] 提供了一项调查，特别侧重于使用深度学习模型分析交通数据。
然而，它只研究交通流预测。一般来说，不同的交通量预测任务有共同的特点，将它们结合起来考虑是有益的。
因此，对交通量预测的研究还缺乏一个广泛而系统的研究。

#### 2.4 贡献

本文首先对现有方法进行分类，描述它们的关键设计选择。

* 本文收集和总结了现有的交通量预测数据集，为其他研究提供了有用的指标。

* 本文进行对比实验研究，以评估不同的模型，确定最有效的组成部分。

* 本文进一步讨论了当前解决方案可能存在的局限性，并列出了未来可能的研究方向。

#### 2.5 现有方法总结

交通预测的研究的方法大致可分为两类：传统方法和基于深度学习的方法。传统方法包括经典统计方法和机器学习方法。
经典的统计方法是根据数据建立一个统计模型来预测和分析数据。最具代表性和最常见的算法是历史平均（HA）、自回归积分移动平均（ARIMA）和向量自回归（VAR）。
然而，这些方法要求数据满足某些假设(例如ARIMA要求时序数据必须是平稳的)，而时变交通量数据太复杂，无法满足这些假设。此外，这些方法只适用于相对较小的数据集，因此在实际应用中，它们的性能通常较差。

随后，针对交通预测问题，提出了支持向量回归（SVR）和随机森林回归（RFR）等多种机器学习方法。这种方法具有处理高维数据和捕捉复杂非线性关系的能力。
然而，这些研究在挖掘复杂时空模式方面的性能仍然有限，因为它们需要领域专家事先设计的额外手工制作的特征，这些特征往往不能完全描述数据的属性，而不是直接从原始数据中学习。

目前效果最好的是基于深度学习的方法，深度学习模型将基本的可学习的块或层堆叠起来形成一个深层架构，整个网络都是端到端的训练。
![Figure 2](../../../../../../img/in-post/2020.07/08/Figure 2.jpg)
Figure 2:Categories of traffic prediction methods.

###  <span id="p3">三、TRADITIONAL METHODS</span>
经典的统计模型和机器学习模型是两种具有代表性的数据驱动交通预测方法。

在时间序列分析中，自回归综合移动平均（ARIMA）及其变体是基于经典统计的最具综合性的方法之一，已被广泛应用于交通预测问题。
然而，这些方法一般都是针对小数据集设计的，不适合处理复杂、动态的时间序列数据。此外，由于通常只考虑时间信息，因此忽略或很少考虑交通数据的空间相关性。

机器学习方法可以对更复杂的数据进行建模，大致分为三类：基于特征的模型、高斯过程模型和状态空间模型。
基于特征的方法通过训练一个基于人工提取交通特征的回归模型来解决交通预测问题。
尽管有这种可行性，基于特征的模型有一个关键的局限性：模型的性能在很大程度上依赖于人工设计的特征。
高斯过程通过不同的核函数对交通数据的内部特征进行建模，需要同时包含空间和时间相关性。
虽然这类方法在交通量预测中被证明是有效可行的，但是它们的计算量和存储压力都比较大，这在大量训练样本的情况下是不合适的。
状态空间模型假设观测值是由马尔科夫隐藏状态生成的。该模型的优点是可以自然地模拟系统的不确定性，更好地捕捉时空数据的潜在结构。
然而，这些模型模拟非线性特征的能力有限，所以，在大多数情况下，它们对于复杂和动态交通数据的建模不是最佳的。

###  <span id="p4">四、DEEP LEARNING METHODS</span>

与传统的学习方法相比，深度学习模型挖掘了更多的特性和复杂的体系结构，能够获得更好的性能。

#### 4.1 建模空间依赖

##### CNN
一系列研究应用CNN从二维时空交通数据中捕捉交通网络中的空间相关性。
由于交通网络很难用二维矩阵来描述，一些研究试图将不同时刻的交通网络结构转换成图像，并将这些图像划分为标准网格，每个网格代表一个区域。
这样，CNNs就可以用来学习不同区域之间的空间特征。
![Figure 3](../../../../../../img/in-post/2020.07/08/Figure 3.jpg)
Figure 3: 2D Convolution.

##### GCN
传统的CNN仅限于对欧式数据进行建模，因此采用GCN对非欧式空间结构数据进行建模，更符合交通路网的结构。
![Figure 4](../../../../../../img/in-post/2020.07/08/Figure 4.jpg)
Figure 4: non-Euclidean spatial structure.

GCN一般包括两类方法，基于谱域的方法和基于空间域的方法。
基于谱域的方法利用傅里叶变换把非欧空间的图转换成欧式空间。
基于空间域的方法利用一种可处理变长邻居结点的卷积核在图上抽取特征。

##### Attention
一条道路的交通状况会受到其他道路的影响。这种影响是高度动态的，随时间而变化。
为了对这些特性进行建模，通常使用空间注意机制来自适应地捕捉道路网络中区域之间的相关性。
其核心思想是在不同的时间步长动态地为不同的区域分配不同的权重。

Table 1 SPATIAL DEPENDENCY MODELING.
![Table 1](../../../../../../img/in-post/2020.07/08/Table 1.jpg)

#### 4.2 建模时间依赖

##### CNN
[7]中介绍了从序列到序列学习的全卷积模型。一项具有代表性的工作流量研究[44]应用纯卷积结构从图形结构的时间序列数据中同时提取时空特征。此外，扩张因果卷积是一种特殊的标准一维卷积。
它通过改变扩张率的大小来调整感受野的大小，有利于捕捉长期的周期依赖关系。
[82]和[75]因此采用扩展因果卷积作为模型的时间卷积层，以捕捉节点的时间趋势。
与递归模型相比，卷积可以为固定大小的上下文创建表示，但是，通过将多个层堆叠在一起，网络的有效上下文大小很容易变得更大。这允许精确控制要建模的依赖项的最大长度。
卷积网络不依赖于前一个时间步长的计算，因此它可以将序列中的每个元素并行化，这样可以更好地利用GPU硬件，更容易优化。
这优于RNN，后者保持过去的整个隐藏状态，防止在一个序列中进行并行计算。

##### RNN
RNN及其变种LSTM或GRU是用于处理序列数据的神经网络。为了模拟交通数据的非线性时间相关性，基于RNN的方法被应用于交通预测[41]。
这些模型依赖于数据的顺序来依次处理数据，因此这些模型的一个缺点是，当对长序列建模时，它们记忆许多时间步之前所学内容的能力可能会下降。

在基于RNN的序列学习中，一种称为编解码器的特殊网络结构被应用于流量预测（[46]、[51]、[54]、[56]–[59]、[73]、[80]、[87]、[88]、[92]、[94]、[95]、[98]、[99]）。
其核心思想是将源序列编码为一个固定长度的向量，并使用解码器生成预测。
编解码结构的一个潜在问题是，无论输入和输出序列的长度如何，编解码之间的语义向量的长度总是固定的，因此当输入信息太长时，会丢失一些信息。

##### Attention

为了解决上述问题，一个重要的扩展就是在时间维度上使用一种注意机制，它可以自适应地选择编码器的相关隐藏状态来产生输出序列。
这类似于空间方法中的注意。这种时间注意机制不仅可以模拟路网中某一位置当前交通状况与以往观测值之间的非线性相关关系，而且可以对长期序列数据进行建模，以解决RNN的不足。

[56]设计了一种时间注意机制，自适应地对不同时间片之间的非线性相关进行建模。[60]结合标准的卷积和注意机制，通过融合相邻时间片上的信息来更新节点的信息，并从语义上表达不同时间片之间的依赖强度。考虑到流量数据是高度周期性的，但不是严格的周期性的，[67]设计了一种周期性转移注意机制来处理长期的周期性依赖和周期性的时间偏移。

##### GCN
[49]不同于大多数流量预测方法，它们分别使用不同类型的神经网络组件来捕捉时空关系，而使用一个组件直接捕捉局部时空关系。具体来说，它首先构造一个包含时间和空间属性的局部时空图，然后使用提出的基于空间的GCN方法同时对时空相关性进行建模。

Table 2 TEMPORAL DEPENDENCY MODELING.
![Table 2](../../../../../../img/in-post/2020.07/08/Table 2.jpg)

#### 4.3 深度学习与传统方法相结合
近年来，越来越多的研究将深度学习与传统方法相结合，并将一些先进的方法应用于交通预测（[106]-[109]）。
这种方法不仅弥补了传统模型非线性表示能力的不足，而且弥补了深度学习方法解释能力差的缺点。
[106]提出了一种基于状态空间生成模型和基于滤波的推理模型的方法，利用深度神经网络实现发射和跃迁模型的非线性，利用递归神经网络实现随时间的依赖。
这种基于非线性网络的参数化为处理任意数据分布提供了灵活性。
[107]提出了一种将矩阵分解方法引入深度学习模型中的深度学习框架，该框架能够对潜在区域函数以及区域间的相关性进行建模，从而进一步提高城市流量预测的模型能力。
[108]开发了一个混合模型，该模型将由时间深度网络正则化的全局矩阵分解模型与捕捉每个维度特定模式的局部深层时间模型相关联。
全局模型和局部模型通过数据驱动的注意机制对每个维度进行组合。因此，可以利用数据的全局模式，并将其与局部校准相结合，以便更好地进行预测。
[109]结合潜在模型和RNN设计了一个网络，用于解决多变量时空时间序列预测问题。该模型捕捉了多个序列在空间和时间层次上的动态性和相关性。

###  <span id="p5">五、THE STATE-OF-THE-ART RESULTS</span>
表3为近期相关应用任务文献的分类，主要集中在中短期预测方面。

Table 3 LITERATURES FOR DIFFERENT TASKS.
![Table 3](../../../../../../img/in-post/2020.07/08/Table 3.jpg)

此外，从这些论文中，本文列出了目前在常用公共数据集下性能最好的方法，如表4所示。
可以得到以下观察：
首先，不同算法的预测性能很大程度上依赖于数据集。
更具体地说，在相同的预测任务下，不同数据集的结果差异很大。
例如，在需求预测任务中，在相同的时间间隔和预测时间下，NYC Taxi和TaxiBJ数据集的精度分别为8.385和17.24。
在预测任务和数据集相同的情况下，性能随着预测时间的增加而下降，如Q-Traffic的速度预测结果所示。
对于同一数据源的数据集，由于选择的时间和区域不同，对精度的影响也较大，如速度预测任务下基于PeMS的相关数据集。
第二，在不同的预测任务中，速度预测任务的准确性总体上可以达到90%以上，明显高于其他接近或超过80%的任务。
因此，这些工作还有很大的改进空间。

一些公司目前正在进行智能交通研究，如amap、滴滴和百度地图。根据2019年amap技术年刊[110]，amap在预测amap驾驶导航的历史速度方面进行了深度学习的探索和实践，它不同于一般的历史平均法，考虑了历史数据呈现的时效性和年周期性特征。结合工业实际，成功地解决了现有的周期性、周期性等特征提取的问题。根据订单数据来衡量某一周的到货时间，不良率为10.1%，比基线低0.9个百分点。

预计到达时间（ETA）、供需和速度预测是滴滴平台的关键技术。DiDi将人工智能技术应用到ETA中，利用神经网络和DiDi的海量订单数据，将MAPE指数降低到11%，实现了在实时大规模请求下为用户提供准确的到达时间期望和多策略路径规划的能力。在预测和调度方面，滴滴采用深度学习模型预测未来一段时间后供需的差异，并提供司机调度服务。对未来30分钟供需缺口的预测精度达到85%。在城市道路速度预测任务中，滴滴提出了基于行驶轨迹标定的预测模型[112]。通过对DiDi-gaia数据集成都和西安数据的对比实验，得出速度预测的总体MSE指标分别降为3.8和3.4。百度通过将辅助信息集成到深度学习技术中，解决了在线路径查询的流量预测任务，并从百度地图上发布了一个包含离线和在线辅助信息的大规模流量预测数据集[88]。在该数据集上，速度预测的总MAPE和2小时MAPE分别下降到8.63%和9.78%。


###  <span id="p6">六、PUBLIC DATASETS</span>

Table 4 STATISTICS PREDICTION FOR DIFFERENT TASKS.
![Table 4](../../../../../../img/in-post/2020.07/08/Table 4.jpg)

高质量的数据集对于准确的交通预测至关重要。在这一部分中，本文全面总结了预测任务所使用的公共数据信息，主要包括两部分：一部分是预测中常用的公共时空序列数据，另一部分是提高预测精度的外部数据。然而，由于不同模型框架的设计或数据的可用性，后一种数据并不是所有模型都使用的。

#### PeMS
它是加州交通局性能测量系统（PeMS）的缩写，它显示在地图上，由39000多个独立探测器实时采集。这些传感器覆盖了加利福尼亚州所有主要都市区的高速公路系统。
可以通过链接[http://pems.dot.ca.gov/](http://pems.dot.ca.gov/)获取。
在此基础上，出现了几种子数据集版本（PeMSD3/4/7(M)/7/8/-SF/-BAY），并得到了广泛的应用。主要区别在于时间和空间的范围，以及数据采集中包含的传感器数量

PeMSD3:此数据集是Song等人处理的一段数据。它包括从2018年9月1日至2018年11月30日的358个传感器和流量信息。可以通过链接[https://github.com/Davidham3/STSGCN](https://github.com/Davidham3/STSGCN)获取

PeMSD4:它描述了旧金山湾区，包含了从2018年1月1日到2018年2月28日的29条道路上的3848个传感器，总共59天。可以通过链接[https://github.com/Davidham3/ASTGCN/tree/master/data/PEMS04](https://github.com/Davidham3/ASTGCN/tree/master/data/PEMS04)获取

PeMSD7(M)：描述加利福尼亚州7区，共有228个站点，时间范围为2012年5月和6月的工作日。可以通过链接[https://github.com/Davidham3/STGCN/tree/master/datasets](https://github.com/Davidham3/STGCN/tree/master/datasets)获取

PeMSD7：这个版本是由Song等人公开发布的。它包含来自883个传感器站的交通流量信息，涵盖2016年7月1日至2016年8月31日。可以通过链接[https://github.com/Davidham3/STSGCN](https://github.com/Davidham3/STSGCN)获取

PeMSD8：它描绘了圣贝纳迪诺地区，包含了从2016年7月1日到2016年8月31日8条道路上的1979个传感器，总共62天。可以通过链接[https://github.com/Davidham3/ASTGCN/tree/master/data/PEMS08](https://github.com/Davidham3/ASTGCN/tree/master/data/PEMS08)获取

PeMSD SF：该数据集描述了旧金山湾区高速公路不同车道的占用率，介于0和1之间。这些测量的时间跨度为2008年1月1日至2009年3月30日，每10分钟采样一次。可以通过链接[http://archive.ics.uci.edu/ml/datasets/PEMS-SF](http://archive.ics.uci.edu/ml/datasets/PEMS-SF)获取

PeMSD-BAY：包含6个月的交通速度统计数据，从2017年1月1日到2017年6月30日，包括湾区325个传感器。可以通过链接[https://github.com/liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)获取

#### METR-LA
它记录了从2012年3月1日到2012年6月30日的四个月的交通速度统计数据，其中包括洛杉矶县高速公路上的207个传感器。
可以通过链接[https://github.com/liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)获取。

#### LOOP
收集自大西雅图地区四条相连高速公路（I-5、I-405、I-90和SR520）上的环路探测器。它包含了2015年全年323个传感器站以5分钟为间隔的交通状态数据。
可以通过链接[https://github.com/zhiyongc/Seattle-Loop-Data](https://github.com/zhiyongc/Seattle-Loop-Data)获取。

#### Los-loop
这个数据集是在洛杉矶县的高速公路上由环路检测器实时采集的。包括207个传感器，其交通速度采集时间为2012年3月1日至2012年3月7日。这些交通速度数据每5分钟汇总一次。
可以通过链接[https://github.com/lehaifeng/T-GCN/tree/master/data](https://github.com/lehaifeng/T-GCN/tree/master/data)获取。

#### TaxiBJ
轨迹数据是北京出租车GPS数据和气象数据，时间间隔为4个时间段：2013年7月1日-2013年10月30日、2014年3月1日至2014年6月30日、2015年3月1日至2015年6月30日、2015年11月1日至2016年4月10日。
可以通过链接[https://github.com/lucktroy/DeepST/tree/master/data/TaxiBJ](https://github.com/lucktroy/DeepST/tree/master/data/TaxiBJ)获取。

#### SZ-taxi
这是2015年1月1日至1月31日深圳市出租车运行轨迹。研究区域包括罗湖区156条主要道路。每条道路上的交通速度每15分钟计算一次。
可以通过链接[https://github.com/lehaifeng/TGCN/tree/master/data](https://github.com/lehaifeng/TGCN/tree/master/data)获取。

#### NYC Bike
自行车的轨迹是从纽约市CitiBike系统收集的。共有约13000辆自行车和800个车站。
可以通过链接[https://www.citibikenyc.com/system-data](https://www.citibikenyc.com/system-data)获取。

#### NYC Taxi
轨迹数据是纽约市2009年至2018年的出租车GPS数据。
可以通过链接[https://www1.nyc.gov/site/tlc/about/tlc-triprecord-data.page](https://www1.nyc.gov/site/tlc/about/tlc-triprecord-data.page)获取。

#### Q-Traffic dataset
它由三个子数据集组成：查询子数据集、交通速度子数据集和路网子数据集。这些数据收集于2017年4月1日至2017年5月31日期间，中国北京的百度地图。
可以通过链接[https://github.com/JingqingZ/BaiduTraffic#Dataset](https://github.com/JingqingZ/BaiduTraffic#Dataset)获取。

#### Chicago
这是2013年至2018年芝加哥共享单车的发展轨迹。
可以通过链接[https://www.divvybikes.com/system-data](https://www.divvybikes.com/system-data)获取。

#### BikeDC
它来自华盛顿的自行车系统。该数据集包括来自472个台站的2011年、2012年、2014年和2016年的四个时间间隔的数据。
可以通过链接[https://www.capitalbikeshare.com/systemdata](https://www.capitalbikeshare.com/systemdata)获取。

#### ENG-HW
它包含了英国政府记录的三个城市之间城际公路的交通流量信息，时间范围为2006年至2014年。
可以通过链接[http://tris.highwaysengland.co.uk/detail/trafficflowdata](http://tris.highwaysengland.co.uk/detail/trafficflowdata)获取。

#### T-Drive
它包含了北京出租车从2015年2月1日到2015年6月2日的大量轨迹。这些轨迹可用于计算每个区域的交通流。
可以通过链接[https://www.microsoft.com/en-us/research/publication/t-drive-driving-directions-based-on-taxi-trajectories/](https://www.microsoft.com/en-us/research/publication/t-drive-driving-directions-based-on-taxi-trajectories/)获取。

#### I-80
2005年4月13日，在加利福尼亚州埃梅里维尔的旧金山湾区收集了详细的I-80型车的轨迹数据。数据集长45分钟，车辆轨迹数据每十分之一秒提供研究区域内每辆车的精确位置。
可以通过链接[https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)获取。

#### DiDi chuxing
滴滴数据开放计划为学术界提供真实和免费的脱敏数据资源。主要包括出行时间指数、多个城市的出行和轨道数据集。
可以通过链接[https://outreach.didichuxing.com/research/opendata/](https://outreach.didichuxing.com/research/opendata/)获取。

#### Travel Time Index data
数据集包括深圳、苏州、济南、海口的出行时间指数，包括市、区、路三级的出行时间指数和平均行驶速度，时间范围为2018年1月1日至2018年12月31日。还包括成都、西安二环路地区滴滴出租汽车平台2018年10月1日至2018年12月1日的运行轨迹数据，以及区域内道路等级、成都和西安城市层面的出行时间指数和平均行驶速度。此外，还包含了成都和西安从2018年1月1日至2018年12月31日的市、区、路两级出行时间指数和平均行驶速度。

#### Common external data
交通量预测往往受到许多复杂因素的影响，这些因素通常被称为外部数据。这里，本文列出常见的外部数据项。
* 气象条件：温度、湿度、风速、能见度和天气状态（晴/雨/风/多云等）
* 驾驶员ID：由于驾驶员的个人情况不同，预测会有一定的影响，因此有必要对驾驶员进行标记，此信息主要用于个人预测。
* 活动：包括各种节假日、交通管制、交通事故、体育赛事、音乐会等活动。
* 时间信息：星期几，一天中的时间片。

###  <span id="p7">七、 FUTURE DIRECTION AND OPEN PROBLEMS</span>

数据量少：大多数现有的解决方案都是数据密集型的。但是，由于各城市的发展水平不均衡，许多城市还存在数据不足的问题。然而，充分的数据通常是深度学习方法的先决条件。解决这个问题的一个可能的方法是使用迁移学习技术来执行跨城市的深层时空预测任务。这项技术旨在有效地将知识从数据丰富的来源城市转移到数据匮乏的目标城市。虽然最近有人提出了一些方法（[51]、[71]、[75]），但这些研究还没有得到深入的研究，例如如何设计一个高质量的数学模型来匹配两个区域，或者如何整合其他可用的辅助数据源等，仍然值得考虑和研究。

知识图融合：知识图是知识集成的重要工具。它是由大量的概念、实体、实体关系和属性组成的复杂关系网络。交通领域知识隐藏在多源、海量的交通大数据中。大规模交通知识图的构建、学习和深度知识搜索有助于挖掘更深层次的交通语义信息，提高预测性能。

长期预测：现有的交通量预测方法主要基于中短期预测，对长期预测的研究很少。长期预测由于时空相关性和不确定性因素的复杂性而变得更加困难。对于长期预测，历史信息可能不会对短期中期预测方法产生太大影响，并且可能需要考虑额外的补充信息。

多源数据：传感器，如环路检测器或摄像头，是当前收集交通数据的主流设备。然而，由于传感器的安装和维护成本昂贵，数据非常稀少。同时，现有的大多数基于以往和当前交通状况的技术并不适用于现实世界的因素，例如交通事故。在大数据时代，交通运输领域产生了大量的数据。在预测交通状况时，可以考虑使用几个不同的数据集。事实上，这些数据是高度相关的。例如，为了提高交通流预测的性能，可以考虑诸如路网结构、交通量数据、兴趣点（poi）和城市人口等信息。多个数据的有效融合可以弥补缺失数据，提高预测精度。

实时预测：实时交通预测的目的是在短时间内进行数据处理和交通状况评估。但是，由于数据量、模型大小和参数的增加，算法运行时间过长，无法保证实时预测的要求。因此，如何设计一个有效的轻量化神经网络，以减少网络计算量，加快网络的运行速度是一个值得研究的课题巨大的挑战。

模型的可解释性：由于神经网络结构复杂，参数量大，算法透明度低，对于神经网络来说，验证其可靠性是众所周知的。缺乏可解释性可能会给交通预测带来潜在的问题。考虑到交通数据的复杂数据类型和表示形式，设计可解释的深度学习模型比其他类型的数据（如图像和文本）更具挑战性。虽然已有的一些工作结合了状态空间模型来提高模型的可解释性（[106]-[109]），但是如何建立一个更具解释性的交通预测深度学习模型还没有得到很好的研究，仍然是一个有待解决的问题。

基准流量预测：随着领域的发展，越来越多的模型被提出，并且这些模型通常以相似的方式呈现。在缺乏统一的实验设置和大数据集的标准化基准的情况下，衡量新的交通预测方法的有效性和比较模型变得越来越困难。此外，模型的设计也变得越来越复杂。虽然大多数方法都进行了烧蚀研究，但还不清楚每个部分是如何改进算法的。因此，利用一个标准的公共数据集设计一个可复制的基准测试框架是非常重要的。

###  <span id="p8">八、 REFERENCES</span>
[1] J. Zhang, Y. Zheng, D. Qi, R. Li, and X. Yi, “Dnn-based prediction
model for spatio-temporal data,” in Proceedings of the 24th ACM
SIGSPATIAL International Conference on Advances in Geographic
Information Systems, 2016, pp. 1–4.

[2] E. Vlahogianni, M. Karlaftis, and J. Golias, “Short-term traffic forecasting: Where we are and where were going,” Transportation Research
Part C Emerging Technologies, vol. 43, no. 1, 2014.

[3] Y. Li and S. Cyrus, “A brief overview of machine learning
methods for short-term traffic forecasting and future directions,”
SIGSPATIAL Special, vol. 10, no. 1, p. 39, 2018. [Online]. Available:
https://doi.org/10.1145/3231541.3231544

[4] A. Singh, A. Shadan, R. Singh, and Ranjeet, “Traffic forecasting,”
International Journal of Scientific Research and Review, vol. 7, no. 3,
2019.

[5] P. Xie, T. Li, J. Liu, S. Du, X. Yang, and J. Zhang, “Urban
flow prediction from spatiotemporal data using machine learning:
A survey,” Information Fusion, 2020. [Online]. Available: https:
//doi.org/10.1016/j.inffus.2020.01.002

[6] B. Williams and L. Hoel, “Modeling and forecasting vehicular traffic
flow as a seasonal arima process: Theoretical basis and empirical
results,” Journal of transportation engineering, vol. 129, no. 6, pp.
664–672, 2003.

[7] E. Zivot and J. Wang, “Vector autoregressive models for multivariate
time series,” Modeling Financial Time Series with S-Plus®, pp. 385–
429, 2006.

[8] R. Chen, C. Liang, W. Hong, and D. Gu, “Forecasting holiday daily
tourist flow based on seasonal support vector regression with adaptive
genetic algorithm,” Applied Soft Computing, vol. 26, pp. 435–443,
2015.

[9] U. Johansson, H. Bostrom, T. L ¨ ofstr ¨ om, and H. Linusson, “Regression ¨
conformal prediction with random forests,” Machine Learning, vol. 97,
no. 1-2, pp. 155–176, 2014.

[10] Y. Lv, Y. Duan, W. Kang, Z. Li, and F. Wang, “Traffic flow prediction
with big data: A deep learning approach,” IEEE Transactions on
Intelligent Transportation Systems, vol. 16, no. 2, pp. 865–873, 2015.

[11] I. Goodfellow, Y. Bengio, and A. Courville, Deep learning. MIT
press, 2016.

[12] N. Kalchbrenner and P. Blunsom, “Recurrent continuous translation
models,” in Proceedings of the 2013 Conference on Empirical Methods
in Natural Language Processing, 2013, pp. 1700–1709.

[13] J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, “Spectral networks and
locally connected networks on graphs,” in Proceedings of International
Conference on Learning Representations, 2014.

[14] D. Rumelhart, G. Hinton, and R. Williams, “Learning representations
by back-propagating errors,” nature, vol. 323, no. 6088, pp. 533–536,
1986.

[15] J. Elman, “Distributed representations, simple recurrent networks, and
grammatical structure,” Machine learning, vol. 7, no. 2-3, pp. 195–225,
1991.

[16] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural
computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[17] K. Cho, B. Van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, ¨
H. Schwenk, and Y. Bengio, “Learning phrase representations using
rnn encoder-decoder for statistical machine translation,” arXiv preprint
arXiv:1406.1078, 2014.

[18] S. Shekhar and B. Williams, “Adaptive seasonal time series models for
forecasting short-term traffic flow,” Transportation Research Record,
vol. 2024, no. 1, pp. 116–125, 2007.

[19] X. Li, G. Pan, Z. Wu, G. Qi, S. Li, D. Zhang, W. Zhang, and Z. Wang,
“Prediction of urban human mobility using large-scale taxi traces and
its applications,” Frontiers of Computer Science, vol. 6, no. 1, pp. 111–
121, 2012.

[20] L. Moreira-Matias, J. Gama, M. Ferreira, J. Mendes-Moreira, and
L. Damas, “Predicting taxi–passenger demand using streaming data,”
IEEE Transactions on Intelligent Transportation Systems, vol. 14, no. 3,
pp. 1393–1402, 2013.

[21] M. Lippi, M. Bertini, and P. Frasconi, “Short-term traffic flow forecasting: An experimental comparison of time-series analysis and supervised
learning,” IEEE Transactions on Intelligent Transportation Systems,
vol. 14, no. 2, pp. 871–882, 2013.

[22] I. Wagner-Muns, I. Guardiola, V. Samaranayke, and W. Kayani, “A
functional data analysis approach to traffic volume forecasting,” IEEE
Transactions on Intelligent Transportation Systems, vol. 19, no. 3, pp.
878–888, 2017.

[23] W. Li, J. Cao, J. Guan, S. Zhou, G. Liang, W. So, and M. Szczecinski,
“A general framework for unmet demand prediction in on-demand
transport services,” IEEE Transactions on Intelligent Transportation
Systems, vol. 20, no. 8, pp. 2820–2830, 2018.

[24] J. Guan, W. Wang, W. Li, and S. Zhou, “A unified framework for
predicting kpis of on-demand transport services,” IEEE Access, vol. 6,
pp. 32 005–32 014, 2018.

[25] Z. Diao, D. Zhang, X. Wang, K. Xie, S. He, X. Lu, and Y. Li, “A hybrid
model for short-term traffic volume prediction in massive transportation
systems,” IEEE Transactions on Intelligent Transportation Systems,
vol. 20, no. 3, pp. 935–946, 2018.

[26] D. Salinas, M. Bohlke-Schneider, L. Callot, R. Medico, and
J. Gasthaus, “High-dimensional multivariate forecasting with lowrank gaussian copula processes,” in Advances in Neural Information
Processing Systems, 2019, pp. 6824–6834.

[27] L. Lin, J. Li, F. Chen, J. Ye, and J. Huai, “Road traffic speed prediction:
a probabilistic model fusing multi-source data,” IEEE Transactions on
Knowledge and Data Engineering, vol. 30, no. 7, pp. 1310–1323, 2017.

[28] P. Duan, G. Mao, W. Liang, and D. Zhang, “A unified spatio-temporal
model for short-term traffic flow prediction,” IEEE Transactions on
Intelligent Transportation Systems, vol. 20, no. 9, pp. 3212–3223, 2018.

[29] H. Tan, Y. Wu, B. Shen, P. Jin, and B. Ran, “Short-term traffic
prediction based on dynamic tensor completion,” IEEE Transactions
on Intelligent Transportation Systems, vol. 17, no. 8, pp. 2123–2133,
2016.

[30] J. Shin and M. Sunwoo, “Vehicle speed prediction using a markov
chain with speed constraints,” IEEE Transactions on Intelligent Transportation Systems, vol. 20, no. 9, pp. 3201–3211, 2018.

[31] K. Ishibashi, S. Harada, and R. Kawahara, “Inferring latent traffic
demand offered to an overloaded link with modeling qos-degradation
effect,” IEICE Transactions on Communications, 2018.

[32] Y. Gong, Z. Li, J. Zhang, W. Liu, Y. Zheng, and C. Kirsch, “Networkwide crowd flow prediction of sydney trains via customized online
non-negative matrix factorization,” in Proceedings of the 27th ACM
International Conference on Information and Knowledge Management,
2018, pp. 1243–1252.

[33] N. Polson and V. Sokolov, “Bayesian particle tracking of traffic flows,”
IEEE Transactions on Intelligent Transportation Systems, vol. 19, no. 2,
pp. 345–356, 2017.

[34] H. Hong, X. Zhou, W. Huang, X. Xing, F. Chen, Y. Lei, K. Bian, and
K. Xie, “Learning common metrics for homogenous tasks in traffic flow
prediction,” in 2015 IEEE 14th International Conference on Machine
Learning and Applications (ICMLA). IEEE, 2015, pp. 1007–1012.

[35] H. Yu, N. Rao, and I. Dhillon, “Temporal regularized matrix factorization for high-dimensional time series prediction,” in Advances in
neural information processing systems, 2016, pp. 847–855.

[36] D. Deng, C. Shahabi, U. Demiryurek, L. Zhu, R. Yu, and Y. Liu,
“Latent space model for road networks to predict time-varying traffic,”
in Proceedings of the 22nd ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining, 2016, pp. 1525–1534.

[37] D. Deng, C. Shahabi, U. Demiryurek, and L. Zhu, “Situation aware
multi-task learning for traffic prediction,” in 2017 IEEE International
Conference on Data Mining (ICDM). IEEE, 2017, pp. 81–90.

[38] A. Kinoshita, A. Takasu, and J. Adachi, “Latent variable model for
weather-aware traffic state analysis,” in International Workshop on
Information Search, Integration, and Personalization. Springer, 2016,
pp. 51–65.

[39] Y. Gong, Z. Li, J. Zhang, W. Liu, and J. Yi, “Potential passenger flow
prediction: A novel study for urban transportation development,” in
Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

[40] Z. Li, N. Sergin, H. Yan, C. Zhang, and F. Tsung‘, “Tensor completion
for weakly-dependent data on graph for metro passenger flow prediction,” in Proceedings of the AAAI Conference on Artificial Intelligence,
2020.

[41] Y. Li and C. Shahabi, “A brief overview of machine learning methods
for short-term traffic forecasting and future directions,” SIGSPATIAL
Special, vol. 10, no. 1, pp. 3–9, 2018.

[42] M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional
neural networks on graphs with fast localized spectral filtering,” in
Advances in neural information processing systems, 2016, pp. 3844–
3852.

[43] T. Kipf and M. Welling, “Semi-supervised classification with graph
convolutional networks,” in Proceedings of the International Conference on Learning Representations, 2017.

[44] B. Yu, H. Yin, and Z. Zhu, “Spatio-temporal graph convolutional
networks: a deep learning framework for traffic forecasting,” in Proceedings of the 27th International Joint Conference on Artificial
Intelligence, 2018, pp. 3634–3640.

[45] X. Geng, Y. Li, L. Wang, L. Zhang, Q. Yang, J. Ye, and Y. Liu,
“Spatiotemporal multi-graph convolution network for ride-hailing demand forecasting,” in Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 33, 2019, pp. 3656–3663.

[46] Y. Li, R. Yu, C. Shahabi, and Y. Liu, “Diffusion convolutional recurrent
neural networks: data-driven traffic forecasting,” in Proceedings of the
International Conference on Learning Representations, 2018.

[47] C. Chen, K. Li, S. Teo, X. Zou, K. Wang, J. Wang, and Z. Zeng, “Gated
residual recurrent graph neural networks for traffic prediction,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33,
2019, pp. 485–492.

[48] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, “Graph wavenet
for deep spatial-temporal graph modeling,” in Proceedings of the 28th
International Joint Conference on Artificial Intelligence. AAAI Press,
2019, pp. 1907–1913.

[49] C. Song, Y. Lin, S. Guo, and H. Wan, “Spatial-temporal sychronous graph convolutional networks: A new framework for spatialtemporal network data forecasting,” https://github.com/wanhuaiyu/
STSGCN/blob/master/paper/AAAI2020-STSGCN.pdf, 2020.

[50] D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by
jointly learning to align and translate,” arXiv preprint arXiv:1409.0473,
2014.

[51] Z. Pan, Y. Liang, W. Wang, Y. Yu, Y. Zheng, and J. Zhang, “Urban
traffic prediction from spatio-temporal data using deep meta learning,”
in Proceedings of the 25th ACM SIGKDD International Conference on
Knowledge Discovery &amp; Data Mining, 2019, pp. 1720–1730.

[52] Y. Li, Z. Zhu, D. Kong, M. Xu, and Y. Zhao, “Learning heterogeneous
spatial-temporal representation for bike-sharing demand prediction,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33,
2019, pp. 1004–1011.

[53] J. Zhang, X. Shi, J. Xie, H. Ma, I. King, and D. Yeung, “Gaan: Gated
attention networks for learning on large and spatiotemporal graphs,”
arXiv preprint arXiv:1803.07294, 2018.

[54] Y. Li and J. Moura, “Forecaster: A graph transformer for forecasting
spatial and time-dependent data,” arXiv preprint arXiv:1909.04019,
2019.

[55] X. Yi, Z. Duan, T. Li, T. Li, J. Zhang, and Y. Zheng, “Citytraffic:
Modeling citywide traffic via neural memorization and generalization
approach,” in Proceedings of the 28th ACM International Conference
on Information and Knowledge Management, 2019, pp. 2665–2671.

[56] C. Zheng, X. Fan, C. Wang, and J. Qi, “Gman: A graph multi-attention
network for traffic prediction,” in Proceedings of the AAAI Conference
on Artificial Intelligence, 2020.

[57] C. Park, C. Lee, H. Bahng, K. Kim, S. Jin, S. Ko, and J. Choo, “Stgrat:
A spatio-temporal graph attention network for traffic forecasting,” arXiv
preprint arXiv:1911.13181, 2019.

[58] X. Geng, L. Zhang, S. Li, Y. Zhang, L. Zhang, L. Wang, Q. Yang,
H. Zhu, and J. Ye, “{CGT}: Clustered graph transformer for
urban spatio-temporal prediction,” in Proceedings of the International
Conference on Learning Representations, 2020. [Online]. Available:
https://openreview.net/forum?id=H1eJAANtvr

[59] X. Shi, H. Qi, Y. Shen, G. Wu, and B. Yin, “A spatial-temporal attention
approach for traffic prediction,” IEEE Transactions on Intelligent
Transportation Systems, pp. 1–10, 2020.

[60] S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan, “Attention based spatialtemporal graph convolutional networks for traffic flow forecasting,” in
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33,
2019, pp. 922–929.

[61] H. Yao, F. Wu, J. Ke, X. Tang, Y. Jia, S. Lu, P. Gong, J. Ye, and
Z. Li, “Deep multi-view spatial-temporal network for taxi demand prediction,” in Thirty-Second AAAI Conference on Artificial Intelligence,
2018.

[62] D. Wang, J. Zhang, W. Cao, J. Li, and Y. Zheng, “When will you arrive?
estimating travel time based on deep neural networks,” in Thirty-Second
AAAI Conference on Artificial Intelligence, 2018.

[63] Z. Lin, J. Feng, Z. Lu, Y. Li, and D. Jin, “Deepstn+: Contextaware spatial-temporal neural network for crowd flow prediction in
metropolis,” in Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 33, 2019, pp. 1020–1027.

[64] P. Zhao, D. Cai, S. Zhang, F. Chen, Z. Zhang, C. Wang, and J. Li,
“Layerwise recurrent autoencoder for general real-world traffic flow
forecasting,” 2018.

[65] Z. Lv, J. Xu, K. Zheng, H. Yin, P. Zhao, and X. Zhou, “Lc-rnn: a
deep learning model for traffic speed prediction,” in Proceedings of the
27th International Joint Conference on Artificial Intelligence, 2018, pp.
3470–3476.

[66] G. Lai, W. Chang, Y. Yang, and H. Liu, “Modeling long-and short-term
temporal patterns with deep neural networks,” in The 41st International
ACM SIGIR Conference on Research & Development in Information
Retrieval, 2018, pp. 95–104.

[67] H. Yao, X. Tang, H. Wei, G. Zheng, and Z. Li, “Revisiting spatialtemporal similarity: A deep learning framework for traffic prediction,”
in Proceedings of the AAAI Conference on Artificial Intelligence,
vol. 33, 2019, pp. 5668–5675.

[68] A. Zonoozi, J. Kim, X. Li, and G. Cong, “Periodic-crn: a convolutional
recurrent model for crowd density prediction with recurring periodic
patterns,” in Proceedings of the 27th International Joint Conference on
Artificial Intelligence, 2018, pp. 3732–3738.

[69] J. Zhang, Y. Zheng, and D. Qi, “Deep spatio-temporal residual networks for citywide crowd flows prediction,” in Thirty-First AAAI
Conference on Artificial Intelligence, 2017.

[70] J. Ke, H. Zheng, H. Yang, and X. Chen, “Short-term forecasting of
passenger demand under on-demand ride services: A spatio-temporal
deep learning approach,” Transportation Research Part C: Emerging
Technologies, vol. 85, pp. 591–608, 2017.

[71] L. Wang, X. Geng, X. Ma, F. Liu, and Q. Yang, “Cross-city transfer
learning for deep spatio-temporal prediction,” Proceedings of the 25th
ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining, 2018.

[72] D. Zang, J. Ling, Z. Wei, K. Tang, and J. Cheng, “Long-term traffic
speed prediction based on multiscale spatio-temporal feature learning
network,” IEEE Transactions on Intelligent Transportation Systems,
vol. 20, no. 10, pp. 3700–3709, 2018.

[73] J. Ye, L. Sun, B. Du, Y. Fu, X. Tong, and H. Xiong, “Co-prediction of
multiple transportation demands based on deep spatio-temporal neural
network,” in Proceedings of the 25th ACM SIGKDD International
Conference on Knowledge Discovery & Data Mining. ACM, 2019,
pp. 305–313.

[74] N. Davis, G. Raina, and K. Jagannathan, “Grids versus graphs:
Partitioning space for improved taxi demand-supply forecasts,” arXiv
preprint arXiv:1902.06515, 2019.

[75] H. Yao, Y. Liu, Y. Wei, X. Tang, and Z. Li, “Learning from multiple
cities: A meta-learning approach for spatial-temporal prediction,” in
The World Wide Web Conference. ACM, 2019, pp. 2181–2191.

[76] S. Guo, Y. Lin, S. Li, Z. Chen, and H. Wan, “Deep spatial–temporal
3d convolutional neural networks for traffic data forecasting,” IEEE
Transactions on Intelligent Transportation Systems, vol. 20, no. 10,
pp. 3913–3926, 2019.

[77] L. Liu, Z. Qiu, G. Li, Q. Wang, W. Ouyang, and L. Lin, “Contextualized spatial–temporal network for taxi origin-destination demand
prediction,” IEEE Transactions on Intelligent Transportation Systems,
vol. 20, no. 10, pp. 3875–3887, 2019.

[78] Z. Zheng, Y. Yang, J. Liu, H. Dai, and Y. Zhang, “Deep and embedded
learning approach for traffic flow prediction in urban informatics,”
IEEE Transactions on Intelligent Transportation Systems, vol. 20,
no. 10, pp. 3927–3939, 2019.

[79] J. Zhang, Y. Zheng, J. Sun, and D. Qi, “Flow prediction in spatiotemporal networks based on multitask deep learning,” IEEE Transactions on Knowledge and Data Engineering, 2019.

[80] R. Jiang, X. Song, D. Huang, X. Song, T. Xia, Z. Cai, Z. Wang,
K. Kim, and R. Shibasaki, “Deepurbanevent: A system for predicting
citywide crowd dynamics at big events,” in Proceedings of the 25th
ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining. ACM, 2019, pp. 2114–2122.

[81] D. Lee, S. Jung, Y. Cheon, D. Kim, and S. You, “Forecasting taxi
demands with fully convolutional networks and temporal guided embedding,” in In Advances in Neural Information Processing Systems,
2018.

[82] S. Fang, Q. Zhang, G. Meng, S. Xiang, and C. Pan, “Gstnet: Global
spatial-temporal network for traffic flow prediction,” in Proceedings of
the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI, 2019, pp. 10–16.

[83] Z. Diao, X. Wang, D. Zhang, Y. Liu, K. Xie, and S. He, “Dynamic spatial-temporal graph convolutional neural networks for traffic
forecasting,” in Proceedings of the AAAI Conference on Artificial
Intelligence, vol. 33, 2019, pp. 890–897.

[84] N. Zhang, X. Guan, J. Cao, X. Wang, and H. Wu, “A hybrid traffic
speed forecasting approach integrating wavelet transform and motifbased graph convolutional recurrent neural network,” arXiv preprint
arXiv:1904.06656, 2019.

[85] M. Wang, B. Lai, Z. Jin, Y. Lin, X. Gong, J. Huang, and X. Hua,
“Dynamic spatio-temporal graph-based cnns for traffic prediction,”
arXiv preprint arXiv:1812.02019, 2018.

[86] Z. Cui, K. Henrickson, R. Ke, and Y. Wang, “Traffic graph convolutional recurrent neural network: A deep learning framework for
network-scale traffic learning and forecasting,” IEEE Transactions on
Intelligent Transportation Systems, 2019.

[87] Z. Zhang, M. Li, X. Lin, Y. Wang, and F. He, “Multistep speed
prediction on traffic networks: A graph convolutional sequence-tosequence learning approach with attention mechanism,” arXiv preprint
arXiv:1810.10237, 2018.

[88] B. Liao, J. Zhang, C. Wu, D. McIlwraith, T. Chen, S. Yang, Y. Guo, and
F. Wu, “Deep sequence learning with auxiliary information for traffic
prediction,” in Proceedings of the 24th ACM SIGKDD International
Conference on Knowledge Discovery & Data Mining. ACM, 2018,
pp. 537–546.

[89] W. Chen, L. Chen, Y. Xie, W. Cao, Y. Gao, and X. Feng, “Multirange attentive bicomponent graph convolutional network for traffic
forecasting,” in Proceedings of the AAAI Conference on Artificial
Intelligence, 2020.

[90] K. Guo, Y. Hu, Z. Qian, H. Liu, K. Zhang, Y. Sun, J. Gao, and
B. Yin, “Optimized graph convolution recurrent neural network for
traffic prediction,” IEEE Transactions on Intelligent Transportation
Systems, pp. 1–12, 2020.

[91] L. Zhao, Y. Song, C. Zhang, Y. Liu, P. Wang, T. Lin, M. Deng, and
H. Li, “T-gcn: A temporal graph convolutional network for traffic
prediction,” IEEE Transactions on Intelligent Transportation Systems,
2019.

[92] L. Bai, L. Yao, S. Kanhere, X. Wang, and Q. Sheng, “Stg2seq: spatialtemporal graph to sequence model for multi-step passenger demand
forecasting,” in Proceedings of the 28th International Joint Conference
on Artificial Intelligence. AAAI Press, 2019, pp. 1981–1987.

[93] K. Lee and W. Rhee, “Graph convolutional modules for traffic forecasting,” arXiv preprint arXiv:1905.12256, 2019.

[94] D. Chai, L. Wang, and Q. Yang, “Bike flow prediction with multi-graph
convolutional networks,” in Proceedings of the 26th ACM SIGSPATIAL
International Conference on Advances in Geographic Information
Systems, 2018, pp. 397–400.

[95] Y. Wang, H. Yin, H. Chen, T. Wo, J. Xu, and K. Zheng, “Origindestination matrix prediction via graph convolution: a new perspective
of passenger demand modeling,” in Proceedings of the 25th ACM
SIGKDD International Conference on Knowledge Discovery & Data
Mining. ACM, 2019, pp. 1227–1235.

[96] Z. Liu, F. Miranda, W. Xiong, J. Yang, Q. Wang, and C. T.Silva,
“Learning geo-contextual embeddings for commuting flow prediction,”
in Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

[97] J. Gehring, M. Auli, D. Grangier, D. Yarats, and Y. Dauphin, “Convolutional sequence to sequence learning,” in Proceedings of the 34th
International Conference on Machine Learning-Volume 70. JMLR.
org, 2017, pp. 1243–1252.

[98] L. Zhu and N. Laptev, “Deep and confident prediction for time series
at uber,” in 2017 IEEE International Conference on Data Mining
Workshops (ICDMW). IEEE, 2017, pp. 103–110.

[99] P. Deshpande and S. Sarawagi, “Streaming adaptation of deep forecasting models using adaptive recurrent units,” in Proceedings of the 25th
ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining, 2019, pp. 1560–1568.

[100] X. Wang, C. Chen, Y. Min, J. He, B. Yang, and Y. Zhang, “Efficient
metropolitan traffic prediction based on graph recurrent neural network,” arXiv preprint arXiv:1811.00740, 2018.

[101] J. Xu, R. Rahmatizadeh, L. Bol¨ oni, and D. Turgut, “Real-time predic- ¨
tion of taxi demand using recurrent neural networks,” IEEE Transactions on Intelligent Transportation Systems, vol. 19, no. 8, pp. 2572–
2581, 2017.

[102] J. Pang, J. Huang, Y. Du, H. Yu, Q. Huang, and B. Yin, “Learning to
predict bus arrival time from heterogeneous measurements via recurrent neural network,” IEEE Transactions on Intelligent Transportation
Systems, vol. 20, no. 9, pp. 3283–3293, 2018.

[103] P. He, G. Jiang, S. Lam, and D. Tang, “Travel-time prediction of
bus journey with multiple bus trips,” IEEE Transactions on Intelligent
Transportation Systems, vol. 20, no. 11, pp. 4192–4205, 2018.

[104] X. Tang, H. Yao, Y. Sun, C. Aggarwal, P. Mitra, and S. Wang, “Joint
modeling of local and global temporal dynamics for multivariate time
series forecasting with missing values,” in Proceedings of the AAAI
Conference on Artificial Intelligence, 2020.

[105] X. Ma, Z. Tao, Y. Wang, H. Yu, and Y. Wang, “Long short-term
memory neural network for traffic speed prediction using remote
microwave sensor data,” Transportation Research Part C: Emerging
Technologies, vol. 54, pp. 187–197, 2015.

[106] L. Li, J. Yan, X. Yang, and Y. Jin, “Learning interpretable deep state
space model for probabilistic time series forecasting,” in Proceedings
of the 28th International Joint Conference on Artificial Intelligence
,
2019, pp. 2901–2908.

[107] Z. Pan, Z. Wang, W. Wang, Y. Yu, J. Zhang, and Y. Zheng, “Matrix
factorization for spatio-temporal neural networks with applications to
urban flow prediction,” in Proceedings of the 28th ACM International
Conference on Information and Knowledge Management, 2019, pp.
2683–2691.

[108] R. Sen, H. Yu, and I. Dhillon, “Think globally, act locally: A deep
neural network approach to high-dimensional time series forecasting,”
in Advances in Neural Information Processing Systems, 2019, pp.
4838–4847.

[109] A. Ziat, E. Delasalles, L. Denoyer, and P. Gallinari, “Spatio-temporal
neural networks for space-time series forecasting and relations discovery,” in 2017 IEEE International Conference on Data Mining (ICDM).
IEEE, 2017, pp. 705–714.

[110] amap technology, “amap technology annual in 2019,” https://files.
alicdn.com/tpsservice/46a2ae997f5ef395a78c9ab751b6d942.pdf.

[111] C. Lea, M. Flynn, R. Vidal, A. Reiter, and G. Hager, “Temporal
convolutional networks for action segmentation and detection,” in
proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, 2017, pp. 156–165.

[112] X. Zhang, L. Xie, Z. Wang, and J. Zhou, “Boosted trajectory calibration
for traffic state estimation,” in 2019 IEEE International Conference on
Data Mining (ICDM). IEEE, 2019, pp. 866–875.

[113] D. Shuman, S. Narang, P. Frossard, A. Ortega, and P. Vandergheynst,
“The emerging field of signal processing on graphs: Extending highdimensional data analysis to networks and other irregular domains,”
IEEE signal processing magazine, vol. 30, no. 3, pp. 83–98, 2013.
