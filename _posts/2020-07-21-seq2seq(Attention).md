---
layout:     post
title:      "从seq2seq到Transformer"
subtitle:   "from seq2seq to Transformer"
date:       2020-07-21
author:     "wumingyao"
header-img: "img/in-post/2020.07/21/bg.jpg"
tags: [seq2seq,Attention,时间依赖模型]
categories: [论文笔记]
---

## 主要内容
* [一、seq2seq](#p1)
* [二、seq2seq+Attention](#p2)
* [三、transformer](#p3)

## 正文

###  <span id="p1">一、seq2seq</span>
#### seq2seq结构
![Figure 0](../../../../../../img/in-post/2020.07/21/figure 0.png)
Figure 0: Illustration of seq2seq

#### seq2seq的缺点
1)中间语义向量无法完全表达整个输入序列的信息。
Encoder 和 Decoder 的唯一联系只有语义编码c,
即将整个输入序列的信息编码成一个固定大小的状态向量再解码，
相当于将信息”有损压缩”。

2)句子X中任意单词对生成某个目标单词yi来说影响力都是相同的，没有任何区别。

3)RNN难以处理长序列的句子。随着输入信息长度的增加，由于向量长度固定，
先前编码好的信息会被后来的信息覆盖，丢失很多信息。

###  <span id="p2">二、seq2seq+Attention</span>

#### seq2seq with Attention模型架构

![Figure 1](../../../../../../img/in-post/2020.07/21/figure 1.png)
Figure 1: Illustration of seq2seq with Attention.

注意力模型就是要从序列中学习到每一个元素的重要程度，然后按重要程度将元素合并。

#### Encoder
![Figure 2](../../../../../../img/in-post/2020.07/21/figure 2.png)
Figure 2: Illustration of encoder.

上图是一个Encoder架构，$s_0$从值上来说与$h_m$是相等的。
首先我们需要将$s_0$和所有的$h_i(i=1,...,m)$计算一个 "相关性"，
比方说计算$s_0$和$h_1$之间的相关性计算得$\alpha_1=align(h_1,s_0)$。
计算得到m个相关性$\alpha_i$之后，将这些值与$h_i$进行加权平均，即

$$c_0=\sum_{i=1}^{m}\alpha_ih_i=\alpha_1h_1+\cdots+\alpha_mh_m$$

![Figure 3](../../../../../../img/in-post/2020.07/21/figure 3.png)

这么做有什么用呢？

因为$c_0$实际上考虑了所有时刻的h,不会像RNN网络那样对长期依赖的捕捉会受到序列长度的限制。
同时根据$\alpha_k$的不同，$c_0$对于不同时刻的关注也不同，$\alpha_k$值越大，
说明在历史序列中的第k个时刻对当前的输出影响越大。

![Figure 4](../../../../../../img/in-post/2020.07/21/figure 4.png)
之后将$s_0,c_0,x_1\'$做为t=0时刻Decoder的输入，计算得到输出$s_1$。

然后再计算$s_1$与所有$h_i(i=1,\cdots,m)$之间的相关性$\alpha_i$,重复上述步骤，直到 Decoder 结束

![Figure 5](../../../../../../img/in-post/2020.07/21/figure 5.png)

![Figure 6](../../../../../../img/in-post/2020.07/21/figure 6.png)

#### 优点
在解码器上应用注意力机制可以使解码器的每个时刻使用不同的背景向量。每个背景向量相当于对输入序列的不同部分分配了不同的注意力

#### 缺点
只关注输入序列和目标序列之间的依赖关系，忽略了输入(目标)序列元素之间的自我依赖关系。
编码器解码器仍然使用RNN，RNN网络那样对长期依赖的捕捉会受到序列长度的限制

没法捕捉位置信息，即没法学习序列中的顺序关系。

###  <span id="p3">三、Transformer</span>
#### Transformer结构
![Figure 7](../../../../../../img/in-post/2020.07/21/figure 7.png)


#### Encoder
![Figure 8](../../../../../../img/in-post/2020.07/21/figure 8.png)
Figure 8：Decoder

Encoder部分是把自然语言序列映射为隐藏层的数学表达的过程。

##### Positional Encoding

因为 Transformer 摈弃了RNN的结构，
因此需要一个东西来标记各个元素之间的时序 or 位置关系，
所以我们必须提供每个元素的位置信息给Transformer，这样它才能识别出语言中的顺序关系。

论文中使用了 sin 和 cos 函数的线性变换来提供给模型位置信息:

$$PE(pos,2i)=sin(pos/10000^{2i/d_{model}}$$

$$PE(pos,2i+1)=cos(pos/10000^{2i/d_{model}}$$

上式中pos指的是一个序列中某个元素的位置，i指的是元素向量的维度序号。

##### self Attention
对于输入的序列$X$,通过wordEmbedding得到该序列中每个元素的向量，并且通过
Positional Encoding得到所有元素的位置向量，将两个向量相加，得到该每个元素
的真实向量表示。第$t$个元素的向量记为$x_t$。

![Figure 9](../../../../../../img/in-post/2020.07/21/figure 9.png)

接着定义三个矩阵$W_Q,W_K,W_V$，使用这三个矩阵分别对所有的元素向量进行线性变换，
于是所有的元素向量又衍生出三个新的向量$q_t,k_t,v_t$。
为了获取第一个元素与其他元素的相关性，需要用第一个元素的查询向量$q_1$乘以其他元素
的键向量$k_t$得到注意力权重，并通过softmax，使他们和为1。
对其它的输入向量也执行相同的操作。
![Figure 10](../../../../../../img/in-post/2020.07/21/figure 10.png)

有了权重之后，将权重其分别乘以对应元素的值向量$v_t$,
最后将这些权重化后的值向量求和，得到第一个输出。
![Figure 11](../../../../../../img/in-post/2020.07/21/figure 11.png)

###### Multi-head Attention
原论文中说到进行 Multi-head Attention 的原因是将模型分为多个头，形成多个子空间，可以让模型去关注不同方面的信息，最后再将各个方面的信息综合起来。其实直观上也可以想到，如果自己设计这样的一个模型，必然也不会只做一次 attention，多次 attention 综合的结果至少能够起到增强模型的作用，也可以类比 CNN 中同时使用多个卷积核的作用，直观上讲，多头的注意力有助于网络捕捉到更丰富的特征 / 信息

##### 残差连接和 Layer Normalization
###### 残差连接
在上一步得到了经过self-attention加权之后输出，也就是self_Attention(Q, K, V)
，然后把他们加起来做残差连接。解决层数变多梯度消失问题。

$$X_{embedding}+self_Attention(Q, K, V)$$

###### Layer Normalization
Layer Normalization 的作用是把神经网络中隐藏层归一为标准正态分布，以起到加快训练速度，加速收敛的作用。

##### Feed forward
Transformer中的feed forward网络可以理解为两个连续的线性变换，这两个变换中间是一个ReLU激活函数.

#### 优点
1)不仅关注输入序列和目标序列之间的依赖关系，还关注了输入(目标)序列元素之间的自我依赖关系。
对长时间依赖特征捕捉能力更强。

2)Transformer并行计算的能力远远超过了 seq2seq 系列模型。
