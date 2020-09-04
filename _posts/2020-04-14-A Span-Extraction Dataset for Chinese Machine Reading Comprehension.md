---
layout:     post
title:      "CMRC 2018 中文数据集"
subtitle:   "Span-Extraction 中文数据集介绍及相应的基线模型"
date:       2020-04-14
author:     "wumingyao"
header-img: "img/in-post/2020.04/14/bg.jpg"
tags: [Span-Extraction,数据集介绍,论文笔记,MRC,NLP,CMRC 2018]
categories: [论文笔记]
---

## 主要内容
* [Abstract](#p1)
* [Introduction](#p2)
* [The Proposed Dataset](#p3)
* [Evaluation Metrics](#p4)
* [Experimental Results](#p5)
* [ Conclusion](#p6)
* [Open Challenge](#p7)

## 参考论文：
[《A Span-Extraction Dataset for Chinese Machine Reading Comprehension》](https://www.aclweb.org/anthology/D19-1600.pdf)

## 正文

###  <span id="p1">Abstract</span>
本文介绍了一个用于中文机器阅读理解的片段抽取任务(Span-Extraction)数据集，以增加该领域的语言多样性。这个数据集由近20000个真实的问题组成，这些问题由人类专家在维基百科的段落中注释。本文还提出了几个基线系统用于证明在这份数据集中的难点。

###  <span id="p2">一、Introduction</span>
阅读和理解自然语言是是实现高级人工智能的关键，机器阅读理解(MRC)的目的是理解所给文章的内容以及根据文章回答相关的问题。各种类型的MRC任务的数据集都已经发表，例如完形填空任务(cloze-style)、片段抽取任务(span-extraction)、自由问答任务(open-domain reading)和多项选择任务(multiple-choice)。

我们也看到了在构建中文机器阅读理解(CMRC)数据集方面的各种努力，完形填空任务(cloze-style)方面的中文数据集有崔一鸣等人提出的：[《人民日报》新闻数据集和儿童童话故事（CFT）数据集](https://www.aclweb.org/anthology/C16-1167.pdf)，为了给数据集增加困难<font color="red">(为啥要给数据增加difficulties呢)</font>，以及自动生成的评估集（开发和测试），他们还发布了一个人工注释的评估集。为了增加多样性和进一步研究迁移学习，崔等人提出另一个[评估数据集](http://www.lrec-conf.org/proceedings/lrec2018/pdf/32.pdf)，该数据集来自儿童阅读材料，该数据集也由人类专家注释，但是查询比完形填空类型更自然,该数据集已用于中国机器阅读理解第一次评估研讨会（CMRC 2017）。在自由回答任务中，提出了一个大规模的开放域中文机器阅读理解数据集([DuReader](https://arxiv.org/abs/1711.05073))，其中包含从搜索引擎上的用户查询日志注释的200k个问题。邵智杰等人提出了CMRC任务的[繁体中文数据集](https://arxiv.org/abs/1806.00920)。

尽管我们已经看到，目前的机器学习方法在[SQuAD数据集](https://arxiv.org/abs/1606.05250)上的表现已经超过了人类，但我们想知道，这些最先进的模型是否也可以在不同语言的数据集上有相同优秀的表现。

![Figure 1](../../../../../../img/in-post/2020.04/14/Figure 1.jpg)
<center>Figure 1: An example of the proposed CMRC 2018 dataset (challenge set). English translation is also given for comparison. </center>
为了进一步促进机器阅读理解研究的发展，本文提出了一个用于汉语机器阅读理解片段提取任务的数据集。Figure 1展示了这份数据集中的一个例子。本文的主要贡献可以总结如下。

* 为了增加阅读理解领域的语言多样性，我们提出了一个包含近20000个人类注释问题的汉语片段抽取阅读理解数据集。
* 为了彻底测试MRC系统的能力，除了开发和测试集之外，还制作了一个挑战集，其中包含了需要在文章中提供各种线索的精心注释的问题。基于BERT的方法只能给出低于50%的F1分数，这说明了它的困难。
* 所提出的中文RC数据，在与SQuAD和其他类似数据集一起研究时，也可以作为跨语言研究的资源。

###  <span id="p3">二、The Proposed Dataset</span>
#### 2.1 Task Definition

一般来说，MRC任务可以描述为一个三元组$\langle P,Q,A \rangle$,其中$P$代表文章，$Q$代表问题，$A$代表答案。对于片段抽取阅读理解任务，问题是由人类注释的，这比完形填空式的MRC数据集要自然得多，答案$A$是从文章$P$中提取的一个片段。这个任务可以通过预测答案片段的开始和结束指针来简化。

#### 2.2 Data Pre-Processing

初始数据集来源于维基百科网页的dump2的中文部分，通过开源工具Wikipedia Extractor3 将原始文件预处理为纯文本,并使用opencc4工具包将繁体字转换为简体字，以达到规范化的目的。

#### 2.3 Human Annotation

在这份数据集中的问题完全是有人类专家构建的，这与之前的采用自动化构建问题的数据集不同。在进行注释之前，一个文档被划分为若干篇文章，并且每篇文章的篇幅不超过500字（使用LTP计数）。注释者首先会评估段文章的适当性，因为有些文章对大部分人来说都是很难理解的，根据以下规则丢弃部分文章。
* 文章内容包含超过30%的非中文字符。
* 包含很多难以理解的专业词汇。
* 包含很多特殊字符和符号。
* 文章是用文言文写的。

在确定文章适合注释后，注释者将阅读文章并根据文章提出问题，并注释一个主要答案，在注释问题的时候，遵循以下原则。
* 每篇文章问题数不超过5个。
* 答案必须是文章中的一个片段，以符合任务定义。
* 鼓励问题多样化，如who/when/where/why/how等。
* 避免直接使用文章中的描述，使用释义或语法转换增加回答困难。
* 长答案(超过30个字)的会被丢弃。

对于评估集，即开发(development)、测试(test)、挑战(chanllenge)，有三个答案可用于更好的评估。除了由问题提出者注释的主要答案外，我们还邀请另外两个注释者为问题写出第二个和第三个答案。在这一阶段，注释者无法看到主要答案，以确保答案不被他人复制，并鼓励答案的多样性。

#### 2.4 Challenge Set

构建该挑战集是为了检验阅读理解模型在处理上下文中需要对各种线索进行综合推理的问题时的能力。挑战集中的问题需要符合以下标准。

![Figure 2](../../../../../../img/in-post/2020.04/14/Figure 2.jpg)
<center>Figure 2:  Question types of the development set. </center>

* 如果答案是一个单词或一个短句，那么答案就不能只由文章中的一个句子来推断。我们鼓励注释者在文章中提出需要综合推理的问题，以增加难度。
* 如果答案属于命名实体的类型或特定类型（如日期、颜色等），则它不能是上下文中的唯一答案，或者机器可以根据其类型轻松地选择答案。例如，如果上下文中只出现一个人名，则它不能用于注释问题。应该至少有两个人的名字，可能会误导机器回答。

#### 2.5 Statistics

数据的预处理统计如Tabel 1所示。

Tabel 1:  Statistics of the CMRC 2018 dataset. (P: Passage, Q: Question, A: Answer)
![Tabel 1](../../../../../../img/in-post/2020.04/14/Tabel 1.jpg)

###  <span id="p4">三、Evaluation Metrics</span>

Table 2: Baseline results and CMRC 2018 participants’ results. Note that, some of the submissions are using
development set for training as well.
![Tabel 2](../../../../../../img/in-post/2020.04/14/Tabel 2.jpg)

在这篇论文中，采用Exact Match和F1 score两个验证指标(参考论文[Rajpurkar等人(2016)](https://arxiv.org/abs/1606.05250))。但是由于中文与英文有很大的不同，所以在使用这两个指标的时候遵循以下原则。公共标点符号和空白符是被忽略不计的。

#### 3.1 EXact Match

测量预测答案和真实答案之间的精确匹配，精确匹配为1，否则得分为0。这与参考论文[Rajpurkar等人(2016)](https://arxiv.org/abs/1606.05250)一致。

#### 3.2 F1-Score

测量预测答案与真实答案之间的词级模糊匹配。我们不把预测答案和真实答案当作”一袋词“，而是计算它们之间最长公共序列（LCS）的长度，并相应地计算F1-score。我们对给定问题的所有真实答案取最大F1。注意，非中文单词不会被分割成字符。

#### 3.3 Estimated Human Performance

本文还测量了人类的表现，以衡量提出的数据集的难度。如前一节所述，开发、测试和挑战集中的每个问题都有三个答案。与[Rajpurkar等人(2016)](https://arxiv.org/abs/1606.05250)测量的方法不同，本文使用交叉验证方法进行测量计算。本文把第一个答案看作是人类预测的答案，把其余的答案当作真实答案。这样，可以通过迭代地将第一、第二和第三个答案视为人类预测答案来获得三个人类预测的表现结果。最后，我们计算三个结果的平均值，作为此数据集上的最终估计人类的表现。

###  <span id="p5">四、Experimental Results</span>

#### 4.1  Baseline System

根据[Devlin等人(2019年)](BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding)，我们采用BERT作为基线系统。为了适应我们的数据集，我们稍微修改run squad.py script$^5$，同时保留大部分原始实现。对于基线系统，我们使用批量大小为32的3e-5初始学习率，并训练两个epochs。文章和问题的最大长度分别设置为512和64。

#### 4.2 Result

结果见Tabel 2,除基线系统外，表中还包括参赛者的CMRC 2018评估结果。训练集和测试集是公开的，通过接受参赛者的提交结果，并评估他们的模型在私有测试集和挑战集的表现，以保持[Rajpurkar等人(2016)](https://arxiv.org/abs/1606.05250)的评估过程的完整性。我们可以看到，大多数参赛者在F1测试中可以获得超过80分的成绩。与F1指标相比，EM指标比SQuAD数据集要低得多（通常在10点以内）。这表明，在汉语机器阅读理解中，如何确定准确的片段边界对提高系统性能起着关键作用。

如Table2最后一栏所示，尽管排名靠前的系统在开发集和测试集上获得了不错的分数，但在挑战集上却未能给出令人满意的结果。然而，正如我们所看到的，在开发、测试和挑战集上估计的人类表现是相对相似的，其中挑战集给出的分数略低。我们还观察到，尽管Z-Reader在测试集上获得了最好的分数，但它在挑战集的EM度量并不是最好的。这表明，目前的阅读理解模式相对不能处理文章中几个线索之间需要综合推理的疑难问题。

基于BERT的方法相对于参与者提交的报告显示出更好的表现。虽然传统模型在测试集中有更高的分数，但是当应用到挑战集时，基于BERT的基线总是更高，这表明BERT提供的丰富的表示有利于解决更难的问题，并且在简单和困难的问题中泛化能力更好。

###  <span id="p6">五、 Conclusion</span>

在这项工作中，我们提出了一个用于中文机器阅读理解的片段提取数据集。该数据集由人类专家进行注释构建，包含近20000个问题以及一个由需要对多条线索进行推理的问题组成的挑战集。评价结果表明，许多模型在开发集和测试集上均能取得优异的成绩，在F1指标中，机器得分仅比人类表现得分低10分。然而，在挑战集方面，模型分数急剧下降，而人的表现与非挑战集几乎相同，这表明在设计更复杂的模型以提高性能方面仍然存在潜在的挑战。我们希望该数据集的发布能在机器阅读理解任务中带来语言多样性，并加速解决需要多线索综合推理的问题的进一步研究。

###  <span id="p7">Open Challenge</span>

可以用CMRC 2018 数据集评估自己的模型，提交模型入口[https://bit.ly/2ZdS8Ct](https://bit.ly/2ZdS8Ct)
