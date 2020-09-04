---
layout:     post
title:      "机器阅读理解综述"
subtitle:   "机器阅读理解的技术和发展趋势"
date:       2020-04-11
author:     "wumingyao"
header-img: "img/in-post/2020.04/11/bg.jpg"
tags:
    - MRC
    - 机器阅读理解
    - Machine Reading Comprehension
    - 论文综述
    - 论文笔记
categories: [论文笔记]
---

## 主要内容
* [一、Introduction](#p1)
* [二、Tasks & Evaluation Metrics](#p2)
* [三、General Architecture](#p3)
* [四、Methods](#p4)
* [五、New Trends](#p5)

## 参考论文：
[《Neural Machine Reading Comprehension:Methods and Trends》](https://arxiv.org/pdf/1907.01118.pdf)

## 正文

### <span id="p1">一、Introduction</span>
**机器阅读理解**（MRC）是一项任务，用于测试机器通过要求机器根据给定的上下文回答问题来理解自然语言的程度。早期的MRC系统是基于规则的，性能非常差。随着深度学习和大规模数据集的兴起，基于深度学习的MRC显著优于基于规则的MRC。常见的MRC任务可以分为四种类型：**完形填空、单项选择、片段抽取、自由回答**。一般的MRC架构由以下几个模块组成：**Embedding、Feature Extraction、Context-Question Interaction、Answer Prediction**。另外，考虑到目前方法的局限性，MRC出现了新的任务，比如，**knowledge-based MRC, MRC with unanswerable questions, multi-passage MRC，conversational question answering**。    
![Figure 1](../../../../../../img/in-post/2020.04/11/figure1.png)
<center>Figure 1: The number of research articles concerned with neural MRC in this survey. </center>

### <span id="p2">二、Tasks & Evaluation Metrics</span>

#### 2.1 Tasks

##### 2.1.1 Cloze Test
给定上下文$C$，从中移除一个词或实体$A(A \in C)$，完形填空任务要求模型通过学习函数$F$使用正确的词或实体进行填空,函数$F$可以表示为$A=F(C-\\{A\\})$，即问题(移除某个词的上下文)与答案的映射。

数据集：CNN & Daily Mail 、CBT、LAMBADA、Who-did-What、CLOTH、CliCR
![Cloze Tests](../../../../../../img/in-post/2020.04/11/Cloze Tests.jpg)
<center>Figure 2: A example of Cloze Tests datasets. </center>

##### 2.1.2 MUltiple Choice
在给定上下文$C$，问题$Q$和候选答案列表$A=\\{A_1,A_2,··,A_n\\}$的情况下，单项选择任务是通过学习函数$F$从$A(A_i\in A)$中选择正确的答案$A_i$，使得$A_i=F（C,Q,A)$。

数据集：MCTest、RACE
![Multiple Choice](../../../../../../img/in-post/2020.04/11/Multiple Choice.jpg)
<center>Figure 3: A example of Multiple Choice datasets. </center>

##### 2.1.3 Span Extraction
尽管完形填空和单项选择一定程度上可以机器阅读理解的能力，但是这两个任务有一定的局限性。首先，单词或实体可能不足以回答问题，需要完整的句子进行回答；其次，在很多情形是没有提供候选答案的。所以片段抽取任务应运而生。

给定包含有$n$个单词的上下文$C$，即$C=\\{t_1,t_2,\cdots,t_n\\}$,与问题$Q$,片段抽取任务要求模型学习函数$F$,来从$C$中抽取连续的子序列$A=\\{t_i,t_{i+1},\cdots,t_{i+k}\\}(1 \leq i+k \leq n)$做为问题$Q$的正确答案，即$A=F(C,Q)$。

数据集：SQuAD、NewsQA、TriviaQA、DuoRC
![Span Extraction](../../../../../../img/in-post/2020.04/11/Span Extraction.jpg)
<center>Figure 4: A example of Span Extraction datasets. </center>

##### 2.1.4 Free Answering
对于答案局限于一段上下文是不现实的，为了回答问题，机器需要在多个上下文中进行推理并总结答案。自由回答任务是四个任务中最复杂的，也更适合现实的应用场景。

给定上下文$C$和问题$Q$，自由应答任务中的正确答案$A$在原始上下文$C$中不一定是子序列，即$A\in C$或$A\notin C$。该任务要求通过学习函数$F$来预测正确答案$A$，即$A=F(C,Q)$

数据集：bAbI、MS MARCO 、SearchQA、NarrativeQA、DuReader
![Free Answering](../../../../../../img/in-post/2020.04/11/Free Answering.jpg)
<center>Figure 5: A example of Free Answering datasets. </center>

#### 2.2 Evaluation Metrics
对于不同的MRC任务，有不同的评估指标。一般有Accuracy，F1 Score，ROUGE-L，BLEU

##### 2.2.1 Accuracy
当给定一个问题集$Q=\\{Q_1,Q_2,\cdots,Q_m\\}$有$m$个问题时，如果模型正确地预测了$n(n<m)$个问题的答案，那么Accuracy计算如下：

$Accuracy=\frac{n}{m}.\tag{1}$

Accuracy指标一般用于**Cloze Test**和**Multiple Choice**任务。

##### 2.2.2 F1 Score
F1 Score是分类任务中常用的度量标准。在MRC方面，候选答案和参考答案都被视为标记袋，真阳性（TP）、假阳性（FP）、真阴性（TN）和假阴性（FN）如Table1所示。

<center>Table 1: The definition of true positive (TP), true negative (TN), false positive (FP), false negative
(FN).</center>
![Table 1](../../../../../../img/in-post/2020.04/11/Table 1.jpg)

**精确率(precision)**和**召回率(recall)** 计算如下:

$precision=\frac{TP}{TP+FP}.\tag{2}$

$recall=\frac{TP}{TP+FN}.\tag{3}$

F1 Score，是精确率和召回率的调和平均值：

$F1=\frac{2\times P\times R}{P+R}.\tag{4}$

其中$P$是$precision$，$R$是$recall$分别表示召回率和准确率。

##### 2.2.3 ROUGE-L
ROUGE (Recall-Oriented Understudy for Gisting Evaluation)是一种用于评估自动摘要好坏的指标，它有多种变体，其中ROUGE-L广泛用于MRC任务，用于衡量候选答案和参考答案的相似性，“L”表示最长公共子序列(LCS)，其计算如下：

$R_{lcs}=\frac{LCS(X,Y)}{m}.\tag{5}$

$P_{lcs}=\frac{LCS(X,Y)}{n}.\tag{6}$

$F_{lcs}=\frac{(1+\beta)^{2}R_{lcs} P_{lcs}}{P_{lcs}+\beta ^{2}P_{lcs}}.\tag{7}$

其中，$X$是长度为$m$个词的真实答案，$Y$是长度为$n$个词的预测答案，$LCS(X,Y)$表示$X$和$Y$的最长公共子序列的长度,所以$R_lcs$和​,$P_lcs$分别表示召回率和准确率，$\beta$用于控制指标的准确率和召回率的重要程度。
​


##### 2.2.4 BLEU
BLEU(Bilingual Evaluation Understudy)，被广泛用于评价翻译的表现(可以通俗的理解为两个句子的相似度)。当适应MRC任务时，BLEU评分衡量预测答案和真实答案之间的相似性。其计算方法如下：

$P_n(C,R)=\frac{\sum_i \sum_k min(h_k(c_i),max(h_k(r_i)))}{\sum_i \sum_k h_k(c_i)}.\tag{8}$

其中$h_k(c_i)$等于第$k$个n-gram出现在候选答案$c_i$中的个数，同理，$h_k(r_i)$等于第$k$个n-gram出现在参考答案$r_i$中的个数。

当预测的候选答案较短时，$P_n(C,R)$值较高，这种精度不能很好地衡量相似性。因此引入惩罚因子BP来缓解这种情况，即BP用来惩罚较短的预测答案，计算公式如下：

$$BP=\begin{cases}
1,l_c>l_r\\
e^{1-\frac{l_r}{l_C}}\\
\end{cases}.\tag{9}
$$

最后，BLEU score 计算如下：

$BLEU=BP\cdot exp(\sum_{n=1}^{N}w_n logP_n).\tag{10}$

### <span id="p3">三、General Architecture</span>

典型的MRC系统以上下文和问题为输入，然后输入答案，系统包含四个关键模块：Embeddings, Feature Extraction, Context-Question Interaction，Answer Prediction。

* **Embeddings**：将单词映射为对应的词向量，可能还会加上POS、NER、question category等信息；
* **Feature Extraction**：抽取question和context的上下文信息，可以通过CNN、RNN等模型结构；
* **Context-Question Interaction**：context和question之间的相关性在预测答案中起着重要作用。有了这些信息，机器就能够找出context中哪些部分对回答question更为重要。为了实现该目标，在该模块中广泛使用attention机制，单向或双向，以强调与query相关的context的部分。为了充分提取它们的相关性，context和question之间的相互作用有时会执行多跳，这模拟了人类理解的重读过程。
* **Answer Prediction**：基于上述模块获得的信息输出最终答案。因为MRC任务根据答案形式分为了很多种，所以该模块与不同任务相关。对于完形填空，该模块输出context中的一个单词或一个实体；对于单项选择，该模块从候选答案中选择正确答案。

![Figure 6](../../../../../../img/in-post/2020.04/11/Figure 6.jpg)
<center>Figure 6: The general architecture of machine reading comprehension system. </center>

### <span id="p4">四、Methods</span>

以下是对MRC系统四大关键模块所使用的方法的介绍。

![Figure 7](../../../../../../img/in-post/2020.04/11/Figure 7.jpg)
<center>Figure 7: Typical techniques in neural MRC systems. </center>

#### 4.1 Embeddings
Embeddings模块将单词转换为对应的向量表示。如何充分编码context和question是本模块中的关键任务。在目前的MRC模型中，词表示方法可以分为传统的词表示和预训练上下文表示。为了编码更丰富的语义信息，MRC系统在原来的词级别表示的基础上，还会融合字向量、POS、NER、词频、问题类别等信息。

(1) Conventional Word Representation
* **One-Hot**：向量长度为词表大小，只有单词的位置为1，其它全0，这种表示方法无法表示两个单词之间的关系；
* **Distributed Word Representation**：将单词编码为连续的低维向量，如word2vec、glove；

(2) Pre-Trained Contextualized Word Representation

尽管分布式词表示可以在编码低维空间中编码单词，并且反映了不同单词之间的相关性，但是它们不能有效地挖掘上下文信息。具体来说，就是词的分布式表示在不同上下文中都是一个常量。为了解决这个问题，研究学者提出了上下文的词表示，在大规模数据集预训练，直接当做传统的词表示来使用或者在特定任务finetune。
* **CoVE**：利用大规模语料训练Seq2Seq模型，将Encoder的输出拿出来作为CoVE。
* **ELMo**：在大规模文本语料预训练双向语言模型，特征抽取模块为LSTM。
* **GPT**：采用单向的Transformer在大规模语料上预训练；
* **BERT**：采用双向的Transformer在大规模语料上预训练，目标任务为masked language model (MLM)和next sentence prediction（NSP）。

(3) Multiple Granularity

Word2Vec或GloVe预先训练的词级嵌入不能编码丰富的句法和语言信息，如词性、词缀和语法，这可能不足以进行深入的机器理解。为了将细粒度的语义信息整合到单词表示中，一些研究人员引入了不同粒度的上下文和问题编码方法。 
* **Character Embeddings**：与词级表示(word-level representations)相比，它们不仅更适合于子词形态的建模，而且可以缓解词汇表外(OOV)问题。
* **Part-of-Speech Tags**：词性（POS）是一类特殊的语法词汇，如名词、形容词或动词。在NLP任务中标记POS标签可以说明单词使用的复杂特征，进而有助于消除歧义。为了将POS标签转换成固定长度的向量，将POS标签作为变量，在训练过程中随机初始化并更新。
* **Name-Entity Tags**：名称实体是信息检索中的一个概念，是指一个真实世界中的对象，如一个人、一个地点或一个组织，它有一个合适的名称。当询问这些对象时，名称实体可能是候选答案。因此，嵌入上下文单词的名称实体标记可以提高答案预测的准确性。名称实体标签的编码方法类似于上述POS标签的编码方法。
* **Binary Feature of Exact Match (EM)**：如果context中的一个单词在query中存在，那么值为1，否则为0；
* **Query-Category**：问题的类型(what,where,who,when,why,how)通常可以提供线索来寻找答案。

##### 4.2 Feature Extraction 

特征提取模块通常放置在嵌入层之后，分别提取上下文和问题的特征。它进一步关注基于嵌入模块所编码的各种句法和语言信息在句子层次上挖掘上下文信息。该模块采用了RNNs、CNNs和Transformer architecture。
###### 4.2.1 Recurrent Neural Networks

在问题方面，双向RNNs的特征提取过程可以分为两类：**词级(word-level)**和**句子级(sentence-level)**。

* 在词级编码中，在第$j$个时间步处嵌入的问题$x_{qj}$的特征提取输出可以表示为：

$Q_j=[\overrightarrow{RNN}(x_{qj});\overleftarrow{RNN}(x_{qj})].\tag{11}$

其中，$\overrightarrow{RNN}(x_{qj})$和$\overleftarrow{RNN}(x_{qj})$分别表示双向rnn的前向和后向隐藏状态,该特征提取过程下图所示：

![Figure 8](../../../../../../img/in-post/2020.04/11/Figure 8.jpg)
<center>Figure 8: Word-level encoding for questions. </center>

* 相比之下，句子级编码将疑问句视为一个整体。特征提取过程可以表示为：

![Figure 9](../../../../../../img/in-post/2020.04/11/Figure 9.jpg)
<center>Figure 9: Sentence-level encoding for questions. </center>

$Q=[\overrightarrow{RNN}(x_{q\|l\|});\overleftarrow{RNN}(x_{q0})].\tag{12}$

其中，$\|l\|$表示问题的长度,$\overrightarrow {RNN}(x_{q\|l\|})$和$\overleftarrow {RNN}(x_{q0})$分别表示双向rnn的前向和后向的最终输出。

因为MRC任务中的上下文通常是一个长句子，所以研究者一般使用词级特征提取法来编码上文信息。和问题在词级编码中类似，在第$j$个时间步处嵌入的上下文$x_{pj}$的特征提取输出可以表示为：

$P_i=[\overrightarrow{RNN}(x_{pi});\overleftarrow{RNN}(x_{pi})].\tag{13}$

###### 4.2.2 Convolution Neural Networks

CNNs广泛应用于计算机视觉领域。应用于NLP任务时，一维cnn在利用滑动窗口挖掘局部上下文信息方面显示出其优越性。在CNNs中，每个卷积层应用不同尺度的特征映射来提取不同窗口大小的局部特征。然后将输出反馈到池化层以降低维数，但最大程度地保留最重要的信息。图10显示了特征提取模块如何使用CNN挖掘问题的本地上下文信息。

![Figure 10](../../../../../../img/in-post/2020.04/11/Figure 10.jpg)
<center>Figure 10: Using CNNs to extract features of question. </center>

如上图所示，给定一个问题$x_q \in \mathbb{R}^{\|l\| \times d}$的词嵌入，其中$\|l\|$代表问题的长度，$d$代表词嵌入的维度，图中卷积层有两个过滤器，带有$k$个输出通道(图中$k=2$)，大小为$f_{t}\times d(\forall{t}=2,3)$，每个过滤器产生一个形状为$(\|l\|-t+1)\times{k}$的feature map,该feature map进一步池化为一个$k$维向量。最终，两个过滤器产生的feature map经过池化后的两个$k$维向量连接为一个2$k$维的向量，用$Q$表示。

尽管n-gram模型和CNNs都可以关注句子的局部特征，但n-gram模型中的训练参数随着词汇量的增加呈指数增长。相比之下，无论词汇量大小如何，CNNs都可以更紧凑、更有效地提取局部信息，因为CNNs不需要表示词汇量中的每一个n-gram。此外，CNNs可以并行训练15次，比RNNs快。CNNs的一个主要缺点是只能提取局部信息，不能处理长序列。

###### 4.2.3 Transformer
Transformer是一个强大的神经网络模型，在各种NLP任务中显示出了良好的性能。与基于RNN或cnn的模型相比，该模型主要基于注意机制，既没有递归，也没有卷积。多个头部的注意力结构不仅在对齐方面有优势，而且可以并行运行。与RNNs相比，Transformer需要更少的训练时间，同时它更关注全局依赖性。但是，如果没有递归和卷积，模型就不能利用序列的位置信息。为了整合位置信息，Transformer添加由正弦和余弦函数计算的位置编码。位置和字嵌入的总和作为输入。图10展示了Transformer架构。在实际应用中，模型通常采用多头自关注和前馈网络来叠加多个块。

![Figure 11](../../../../../../img/in-post/2020.04/11/Figure 11.jpg)
<center>Figure 11: Using the Transformer to extract features of question. </center>

##### 4.3 Context-Question Interaction

通过提取context和question之间的相关性，模型能够找到答案预测的证据。根据模型是如何抽取相关性的方式，目前的工作可以分为两类，一跳交互和多条交互。无论哪种交互方式，在MRC模型中，attention机制在强调context哪部分信息在回答问题方面更重要发挥着关键作用。在机器阅读理解中，attention机制可以分为无向和双向的。

###### 4.3.1 Unidirectional Attention

单向的attention主要是根据问题关注context中最相关的部分。如果context中的单词与问题更相似，那么该单词更可能是答案。通过计算公式$S_i=f(P_i,Q)$得到context中的单词与question的相似度，其中$f(\cdot)$表示计算相似度的函数，$P_i$是context中的单词的embedding，$Q$是question的句子表示，最后通过softmax进行权重归一化，获得上下文中每个词的注意力权重$\alpha_i$:

$\alpha_i=\frac{expS_i}{\sum_{j}expS_j}\tag{14}$

![Figure 12](../../../../../../img/in-post/2020.04/11/Figure 12.jpg)
<center>Figure 12: Using unidirectional attention to mine correlation between the context and question. </center>

不同的模型对$f(\cdot)$有不同的选择，一般有以下两种：

$S_i=tanh(W_{P}P_{i}+W_{Q}Q.\tag{15}$

$S_i=Q^{T}W_{S}P_{i}.\tag{16}$

其中，$W_P,W_Q,W_S$是训练可得的参数。

单向的attention可以关注context中最重要的词来回答问题，但是该方法无法关注对答案预测也至关重要的question的词。因此，单向的attention不足以抽取context和query之间的交互信息。

###### 4.3.2 Bidirectional Attention

同时计算query-to-context attention和context-to-query attention

图13显示了计算Bidirectional Attention的过程。首先，通过计算上下文语义嵌入$P_i$和问题语义嵌入$Q_j$之间的匹配分数，得到成对匹配矩阵$M(i,j)$。然后，column-wise SoftMax函数的输出可以被视为query-to-context注意力的权重，用$\alpha$表示，row-wise SoftMax函数的输出表示context-to-query注意力权重$\beta$。

![Figure 13](../../../../../../img/in-post/2020.04/11/Figure 13.jpg)
<center>Figure 13: Using bidirectional attention to mine correlation between the context and question. </center>

##### 4.3.3 One-Hop Interaction

单跳交互是一种浅层架构，上下文和问题之间的交互只计算一次。虽然这种方法可以很好地处理简单的完形填空测试，但当问题需要在上下文中对多个句子进行推理时，这种单跳交互方法很难预测正确答案。

##### 4.3.4 Multi-Hop Interaction

Multi-Hop Interaction可以记住之前的context和question信息，能够深度提取相关性并聚合答案预测的证据。

#### 4.4 Answer Prediction 

该模块与任务高度相关，之前我们将MRC分为四类，分别是完形填空、单项选择、片段抽取、自由回答，那么对应的答案预测方法也有四种，分别是word predictor，option selector，span extractor，answer generator。

##### 4.4.1 Word Predictor

完形填空要求模型预测单词或实体进行填空，该单词或实体来自给定的context。这方面的工作有Attentive Reader、Attention Sum Reader。

##### 4.4.2 Option Selector

对于多选任务，模型从候选答案列表中选择一个正确答案。很普遍的做法是衡量attentive context representations和候选答案表示之间的相似度，选择相似度最高的作为预测答案。

##### 4.4.3 Span Extractor

片段抽取任务是完形填空任务的扩展，要求模型从context中抽取一个子串，而不是一个单词。目前的工作有Sequence Model、Boundary Model。

##### 4.4.4 Answer Generator

自由回答任务中答案不再局限于context中的一个片段，而是需要根据context和question合成答案。目前的工作有S-Net。

#### 4.5 Additional Tricks

##### 4.5.1 Reinforcement Learning
##### 4.5.2 Answer Ranker 
##### 4.5.3 Sentence Selector

实际上，如果给MRC模型一个很长的文档，那么理解全部上下文来回答问题是很费时的。但是，事先找到问题中最相关的句子是加速后续训练过程的一种可能方法。有研究学者提出了sentence selector来选择回答问题需要的最小句子集合。

### <span id="p5">五、New Trends</span>

#### 5.1 Knowledge-Based Machine Reading Comprehension 

有时候，我们只根据context是无法回答问题的，需要借助外部知识。因此，基于外部知识的MRC应运而生。KBMRC和MRC的不同主要在输入部分，MRC的输入是context和question，而KBMRC的输入是context、question、knowledge。

<center>Table 2: Some Examples in KBMRC </center>

![Table 2](../../../../../../img/in-post/2020.04/11/Table 2.jpg)

目前KBMRC的主要挑战在于：
* Relevant External Knowledge Retrieval

知识库中存储着各种各样的知识，实体有时可能因为多义词而产生误导，例如，“苹果”可以指水果或公司。抽取与上下文和问题密切相关的知识决定了基于知识的答案预测的性能。

* External Knowledge Integration

与语境中的文本和问题相比，外部知识库中的知识有其独特的结构。如何对这些知识进行编码，并将其与上下文和问题的表示结合起来，仍然是一个正在进行的研究挑战。

#### 5.2 Unanswerable Questions 

有一个潜在的假设就是MRC任务中正确答案总是存在于给定的上下文中。显然这是不现实的，上下文覆盖的知识是有限的，存在一些问题是无法只根据上下文就可以回答的。因此，MRC系统应该区分这些无法回答的问题。

<center>Table 3: Unanswerable question example in SQuAD 2.0 </center>

![Table 3](../../../../../../img/in-post/2020.04/11/Table 3.jpg)

关于不可回答的问题，相比传统的MRC，在该新任务上又有新的挑战：

* Unanswerable Question Detection

模型应该知道它不知道的东西。在通过文章理解问题和推理之后，MRC模型应该根据给定的上下文判断哪些问题是不可能回答的，并将它们标记为不可回答的。

* Plausible Answer Discrimination 

为了避免假答案的影响，MRC模型必须验证预测的答案，并从正确的答案中说出可信的答案。

##### 5.3 Multi-Passage Machine Reading Comprehension 

在MRC任务中，相关的段落是预定义好的，这与人类的问答流程矛盾。因为人们通常先提出一个问题，然后再去找所有相关的段落，最后在这些段落中找答案。因此研究学者提出了multi-passage machine reading comprehension，相关数据集有MS MARCO、TriviaQA、SearchQA、Dureader、QUASAR。

##### 5.4 Conversational Question Answering 

MRC系统理解了给定段落的语义后回答问题，问题之间是相互独立的。然而，人们获取知识的最自然方式是通过一系列相互关联的问答过程。比如，给定一个问答，A提问题，B回复答案，然后A根据答案继续提问题。这个方式有点类似多轮对话。
