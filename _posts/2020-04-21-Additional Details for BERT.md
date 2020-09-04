---
layout:     post
title:      "Bert 模型的相关细节"
subtitle:   "Bert 模型的理解"
date:       2020-04-21
author:     "wumingyao"
header-img: "img/in-post/2020.04/21/bg.jpg"
tags: [Span-Extraction,Bert,论文笔记,MRC,NLP]
categories: [论文笔记]
---

## 主要内容
* [Additional Details for BERT](#p1)
* [Detailed Experimental Setup](#p2)
* [Related Work](#p3)
* [Bert](#p4)
* [Experiments](#p5)
* [Ablation Studies](#p6)
* [ Conclusion](#p7)

## 参考论文：
[《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](https://arxiv.org/pdf/1810.04805.pdf)

## 正文

###  <span id="p1">A Additional Details for BERT</span>
#### A.1 Illustration of the Pre-training Tasks

<strong>Masked LM and the Masking Procedure</strong> 假设无标签句子是my dog is hairy,在随机mask的过程中，选择了第四个词hairy,则masking 程序可以有如下处理：
* 80%的可能性用[MASK]来代替被mask的词，即my dog is hairy $\rightarrow$ my dog is [MASK]
* 10%的可能用一个随机的单词来替换，即my dog is hairy $\rightarrow$ my dog is apple
* 10%的可能保持该词不变，即my dog is hairy $\rightarrow$ my dog is hairy.这样做的目的是使表达偏向于实际观察到的单词。<font color='red'>(啥意思)</font>

这个过程的优点是，Transformer编码器不知道它将被要求预测哪些单词，或者哪些单词已被随机单词替换，因此它必须保持每个输入单词的分布式上下文表示。
![Figure 1](../../../../../../img/in-post/2020.04/21/Figure 1.jpg)
Figure 1: Differences in pre-training model architectures. BERT uses a bidirectional Transformer. OpenAI GPT
uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and right-toleft LSTMs to generate features for downstream tasks. Among the three, only BERT representations are jointly
conditioned on both left and right context in all layers. In addition to the architecture differences, BERT and
OpenAI GPT are fine-tuning approaches, while ELMo is a feature-based approach.

<strong>Next Sentence Prediction</strong> NSP任务描述如下：
![Figure 2](../../../../../../img/in-post/2020.04/21/Figure 2.jpg)

#### A.2 Pre-training Procedure

为了生成每个训练输入序列，作者从语料库中抽取两段文本，并称之为“句子”，尽管它们通常比单个句子长得多（但也可以更短）。第一句接受A嵌入，第二句接受B嵌入。50%的可能A是B的下一句，50%的可能A是随机的句子，选出两个句子组合后词长度$\leq$512的样本。LM masking是在wordpiece标记化之后应用的，掩蔽率为15%。

作者设置batch_size=256,steps=1000000,epochs=40，所用的语料库超过33亿个词。$lr=le-4,\beta_1=0.9,\beta_2=0.999$，L2权重衰减0.01，学习率在前10000步预热，学习率呈线性衰减。训练损失是真实的下一句和预测的下一句的似然值的均值和。

#### A.3 Fine-tuning Procedure

对于微调，除了批量大小、学习速率和训练阶段数之外，大多数模型超参数与预训练中的相同。辍学率始终保持在0.1。最佳超参数值是特定于任务的，但在实验中发现，以下参数的设定适用于所有任务：
* Batch size：16,32
* Learning rate(Adam)：5e-5,3e-4,2e-5
* Number of epochs：2,3,4

###  <span id="p1">B Detailed Experimental Setup</span>
![Figure 3](../../../../../../img/in-post/2020.04/21/Figure 3.jpg)
Figure 3: Illustrations of Fine-tuning BERT on Different Tasks.

使用BETT实现其他NLP任务如Figure 3所示。
