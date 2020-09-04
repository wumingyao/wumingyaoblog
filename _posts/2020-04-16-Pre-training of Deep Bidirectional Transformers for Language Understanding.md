---
layout:     post
title:      "Bert 论文阅读笔记"
subtitle:   "Bert 模型的理解"
date:       2020-04-16
author:     "wumingyao"
header-img: "img/in-post/2020.04/16/bg.jpg"
tags: [Span-Extraction,Bert,论文笔记,MRC,NLP]
categories: [论文笔记]
---

## 主要内容
* [Abstract](#p1)
* [Introduction](#p2)
* [Related Work](#p3)
* [Bert](#p4)
* [Experiments](#p5)
* [Ablation Studies](#p6)
* [ Conclusion](#p7)

## 参考论文：
[《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》](https://arxiv.org/pdf/1810.04805.pdf)

## 正文

###  <span id="p1">Abstract</span>
这篇论文介绍一个新的语言表达模型BERT(Bidirectional Encoder Representations from Transformers)。BERT 是用于语言理解的预训练深度双向编码表征的 transformer结构。它被设计为通过在所有网络层中基于左右文本来预训练深度双向表征。因此通过外接一个输出层来 fine-tuned 预训练好的BERT 表征形成一个新的模型，这种做法可以将BERT运用在大量的其他任务上，例如问题回答任务、语言推理任务等。

Bert模型易理解且功能强大，它在11个NLP任务中都表现的最好，在机器阅读理解SQuAD1.1跑出的成绩，在两个指标上全面超越人类。GLUE基准80.04%（7.6%绝对提升），MultiNLI准确率86.7%（5.6%绝对提升）。

###  <span id="p2">一、Introduction</span>
语言模型的预训练对于改善许多自然语言处理任务是有效的。这些任务包括句子级别的任务像自然语言推理([Bowman等人(2015)](https://arxiv.org/pdf/1508.05326.pdf);[Williams等人(2018)](https://arxiv.org/pdf/1704.05426.pdf)),和释义。句子级任务目的是通过对句子的整体分析来预测句子之间的关系。

目前存在的将预训练好的语言表征运用在下游任务的方法主要有：基于特征（feature-based）的和微调（fine-tuning）。基于特征的方法，比如[ELMo](https://www.aclweb.org/anthology/W18-5400.pdf)，将预训练好的表征作为额外的特征加到某个基于任务的架构中。基于微调的方法，比如 [OpenAI GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)，引入最小任务相关的参数，运用下游的特定任务来微调训练好的参数。这两种方法在训练前都有相同的目标函数，即使用单向语言模型来学习一般的语言表征。目前的这些方法限制了预训练模型的能力特别是基于微调的方法。最主要的限制在于传统的语言模型都是单向的，这种设计限制了训练时模型结构的选择。例如，OpenAI GPT 使用了一种从左向右的架构，在Transformers 的 self-attention layer 中每个分词只能被添加到它前一个分词之后。这种设计对于处理句子级的任务来说是个 sub-optimal，但是在分词层面的任务中微调训练好的模型时，这种设计可能带来灾难性的结果，因为在这类任务中，从两个方向合并上下文是至关重要的。

作者引入了 BERT通过提出了一个新的预训练目标：“masked language model”（MLM）。遮蔽语言模型随机地遮蔽输入中的一些分词，目标是为了能够在仅基于上下文的情况下预测出被遮蔽的分词的id。不同于以往的从左至右的预训练语言模型，MLM模型允许融合左边和右边两边的上下文，从而可以形成深度双向的Transformer。同时文章引入了预测下一个句子的任务来预训练文本对(text-pair)表征。

这篇文章的贡献：
* 证实了双向预训练对于语言表征的重要性
* 证实了预训练表征能够减少了许多高度工程化的特定任务特征架构的需求<font color="red">（啥意思）</font>
* BERT 在11项自然语言处理任务中取得了最先进的效果，bert的实现代码：[https://github.com/google-research/bert](https://github.com/google-research/bert).

###  <span id="p3">二、Related Work</span>
语言表征的预训练由来已久，以下简要回顾了最广泛使用的方法。
#### 2.1 Unsupervised Feature-based Approaches
预训练的词嵌入方法是NLP系统的组成部分，与从零开始训练的方法相比，与训练方法有更好的表现。为了预训练单词嵌入向量，[Mnih和Hinton等人(2009)](http://papers.nips.cc/paper/3583-a-scalable-hierarchical-distributed-language-model.pdf)使用了从左到右的语言建模目标，[Mikolov等人(2013)](https://arxiv.org/pdf/1310.4546.pdf)的模型在左右上下文中区分正确和错误单词。

这些方法已经推广到更广的粒度，例如句子级的嵌入或者段落级嵌入。为了训练句子表示，先前的工作通过训练目标为‘对候选的下一个句子进行排序’的模型获得（[Jernite等人(2017)](http://arxiv.org/abs/1705.00557)和[Logeswaran和Lee(2018)](https://openreview.net/forum?id=rJvJXZb0W)），从左到右生成下一个句子单词以表示前一个句子（[Kiros等人(2015)](https://arxiv.org/pdf/1506.06726)），或去噪自编码器派生的目标（[Hill等人(2016)](https://arxiv.org/pdf/1602.03483)）。

ELMo及其前身（Peters等人([2017](https://arxiv.org/pdf/1705.00108)，[2018a](https://arxiv.org/pdf/1802.05365)))从不同的维度概括了传统的单词嵌入研究。它们从左到右和从右到左的语言模型中提取上下文相关的特征。每个词的上下文表示是从左到右和从右到左表示的连接。当将上下文单词嵌入与现有的特定任务架构相结合时，ELMo提出几个主要NLP基准（[Peters等人(2018a)](https://arxiv.org/pdf/1802.05365)）的最新技术，包括问答[Rajpurkar等人(2016)](https://arxiv.org/abs/1606.05250)、情感分析[Socher等人(2013)](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)和命名实体识别[Tjong等人(2003)](https://dl.acm.org/doi/pdf/10.3115/1118853.1118872?download=true)。[Melamud等人(2016)](https://arxiv.org/pdf/1810.04805.pdf#page=11&zoom=100,0,498)建议通过一项任务学习语境表征，使用LSTMs从左右语境预测单个单词。与ELMo相似，它们的模型是基于特征的，并且没有深度的双向性。[Fedus等人(2018)](https://arxiv.org/abs/1801.07736)表明完形填空任务可以用来提高文本生成模型的健壮性。

#### 2.2 Unsupervised Fine-tuning Approaches

与基于特征的方法一样，第一种方法是在这个方向上只从未标记的文本中预训练单词嵌入参数。
最近，产生上下文词级表示的句子或文档编码器已经从未标记的文本中进行了预训练，并针对有监督的下游任务进行了微调。这些方法的优点是几乎不需要从头学习参数。至少在一定程度上是由于这个优势，OpenAI-GPT在GLUE基准测试的许多句子级任务上取得了先前最先进的结果。

![Figure 1](../../../../../../img/in-post/2020.04/16/Figure 1.jpg)
<center>Figure 1: Overall pre-training and fine-tuning procedures for BERT. Apart from output layers, the same architectures are used in both pre-training and fine-tuning. The same pre-trained model parameters are used to initialize models for different down-stream tasks. During fine-tuning, all parameters are fine-tuned. [CLS] is a special symbol added in front of every input example, and [SEP] is a special separator token (e.g. separating questions/answers). </center>

#### 2.3 Transfer Learning from Supervised Data

也有研究显示，在大数据集的监督任务中，如自然语言推理和机器翻译，可以有效地进行转换.计算机视觉研究也证明了从大的预先训练的模型中转移学习的重要性，其中一个有效的方法是用ImageNet对预先训练的模型进行微调。

###  <span id="p4">三、Bert</span>

BERT架构中有两个步骤：预训练(pre-training)和微调(fine-tuning)。在预训练阶段，BERT模型在不同的预训练任务中对未标记数据进行训练。对于微调阶段，首先使用预先训练的参数初始化BERT模型，然后使用来自下游任务的标记数据对所有参数进行微调整。每个下游任务都有单独的微调模型，即使它们是用相同的预训练参数初始化的。

BERT的一个显著特点是其在不同任务中的统一架构。预训练的架构和最终的下游任务的架构之间的差别很小。

BERT的模型结构是基于[Vaswani等人(2017)](https://arxiv.org/pdf/1706.03762.pdf)实现的多层双向变压器编码器,该模型已经发布在tensor2tensor库中。

BERT的参数说明：$L$表示层数(Transformer block 的数量)，$H$表示隐藏层的数量，$A$表示self-attention heads 的数量。主要的两个模型的大小分别为：
* BERT$_{BASE}$($L=12,H=768,A=12$,Total Parameters=110M)
* BERT$_{LARGE}$($L=24,H=1024,A=16$,Total Parameters=340M)

为了进行比较，作者选择了与OpenAI GPT具有相同模型大小的BERT$_{BASE}$。BERT Transformer使用双向self-attention，而GPT Transformer使用约束的self-attention，其中每个词只能关注其左侧的上下文。

为了使BERT能够处理各种下游任务，BERT的输入能够在一个词序列中明确地表示一个句子和一对句子(如$\langle Question，Answeri \rangle$)。句子可以是文章中的任意片段。“sequence”是指输入到BERT的词序列，它可以是一个句子，也可以是打包在一起的两个句子。

作者使用了具有30000个词的词汇表进行[WordPice (武永辉等人2016)](https://arxiv.org/abs/1609.08144)词嵌入。每个序列的第一个词总是一个特殊的分类标记([CLS])。与此标记对应的最终隐藏状态用作分类任务的聚合序列表示。句子对被打包成一个序列，有两种方法可以区分句子。第一种是用一个标记词([SEP])来划分这个句子对，第二种是增加一个可学习的嵌入到每个词中，用于区分该词属于句子A还是句子B。如Figure 1所示，$E$表示输入嵌入，$C \in\mathbb{R}^H $表示[CLS]标记词的最终隐藏向量，第$i$个输入词的最终隐藏向量表示为$T_i \in \mathbb{R}^H$。

对于给定的词，其输入表示是通过对相应的词、片段和位置嵌入求和来构造的。这种构造的可视化如Figure 2所示。

![Figure 2](../../../../../../img/in-post/2020.04/16/Figure 2.jpg)
<center>Figure 2: BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings. </center>

#### 3.1 Pre-training BERT

<strong>Task #1: Masked LM</strong>  作者认为深度双向模型比从左到右模型或从左到右模型和从右到左模型的浅层连接更强大。传统的模型都是从左到右或者从右到左训练，这极大地限制了模型的能力。在多层的双向训练中会出现“标签泄漏(since bidirectional conditioning would allow each word to indirectly “see itself”)，作者提出 Masked LM 方法。对输入的句子随机地 Mask 住 15%的 分词，训练模型去预测句子中被 Mask的分词。被 Mask的分词对应的最后的向量会被传入到输出的 Softmax函数。

这种方法虽然可以得到双向的预训练模型，但是也存在两个问题。

第一个就是 pre-training 和 fine-tunning 语料不匹配的问题，因为 被 Mask住的分词不会在 fine-tunning阶段出现。为了解决这个问题，被随机挑选出来被 Mask 的分词比不总是以 [MASK]出现。

* 80%的时间，用 [MASK]代替被选中的词，如 my dog is hairy  -> my dog is [MASK]
* 10%的时间，用一个随机挑选的词代替被选中的词，如 my dog is hairy -> my dog is apple
* 10%的时间，保持原来的单词不变，如 my dog is hairy -> my dog is hairy

Transformer解码器不知道哪个单词需要被预测以及哪个单词是被替换了的，因此它被要求保持对每个输入token分布式的表征，否则会一直记着被 Mask的词是 hairy。

第二个问题就是因为每个Batch中只有15%的分词被预测，所以模型需要训练更多的次数才能收敛。但是模型带来的性能提升比计算消耗更值得。

<strong>Task #2: Next Sentence Prediction (NSP)</strong>  许多自然语言处理的任务如QA、NLI，是基于模型对两个不同文本之间关系的理解的，而语言模型并不能直接反应这种关系。为了使预训练模型能够很好地处理这类任务，作者提出了 next sentence prediction 任务。特别地，在训练集中，有50%的句子B是句子A的真正的下一句(labeled as IsNext)，而另外50%的句子B是从语料中随机抽取的句子(labeled as NotNext)。实验结果表明，增加的这个任务在 QA 和 NLI 都取得了很好的效果。

<strong>Input</strong>=[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]

<strong>LABEL</strong>=IsNext

<strong>Input</strong>=[CLS] the man went to [MASK] store [SEP] penguin [MASK] are flight ##less birds [SEP]

<strong>LABEL</strong>=NotNext

<strong>Pre-training data</strong> 预训练过程中使用的语料是 BooksCorpus (800M 词) 和 English Wikipedia (2,500M 词).

#### 3.2 Fine-tuning BERT

Transformer中的self-attention机制允许BERT通过调整适当的输入和输出，对许多下游任务进行建模，无论这些任务涉及单个文本还是文本对。

对于每个任务，只需将特定于任务的输入和输出插入到BERT中，并对所有参数进行端到端的微调。在输入部分，在预训练阶段的句子A和句子B类似于(1)在段落中的句子对，(2)蕴含中的前提-假设对，(3)问题回答任务中的问题-文章对，(4)文本分类中的text-∅对。在输出部分，词级表示被输入到词级任务的输出层，例如序列标记或问题回答，[CLS]表示被输入到分类的输出层，例如蕴涵或情感分析。

###  <span id="p5">四、Experiments</span>

本节将介绍11个NLP任务的BERT微调结果。

GLUE(The General Language Understanding Evaluation)基准是多种自然语言理解的集合任务，是一个用于评估通用 NLP 模型的基准，其排行榜可以在一定程度上反映 NLP 模型性能的高低。（GLUE 基准针对的是英文任务）。

对于GLUE的微调，输入序列如上文所描述的一样（单个句子或者句子对），用隐藏向量$C \in \mathbb{R}^H$做为标记词([CLS])的输出的聚合表示。在微调阶段新引进来的参数仅有分类层的权重$W \in \mathbb{R}^{K \times H}$，其中$K$是label的数量。使用$C$和$W$来计算分类loss，例如$log(softmax(CW^T)$。


![Table 1](../../../../../../img/in-post/2020.04/16/Table 1.jpg)
Table 1: GLUE Test results, scored by the evaluation server (https://gluebenchmark.com/leaderboard).
The number below each task denotes the number of training examples. The “Average” column is slightly different
than the official GLUE score, since we exclude the problematic WNLI set.8 BERT and OpenAI GPT are singlemodel, single task. F1 scores are reported for QQP and MRPC, Spearman correlations are reported for STS-B, and
accuracy scores are reported for the other tasks. We exclude entries that use BERT as one of their components.

作者设置batch_size=32，epochs=3,并对表格中的GLUE任务的数据进行微调。对于每个任务，在开发集中选择了(5e-5,4e-5,3e-5,2e-5中)最优的学习率。

结果如Table 1所示，$BERT_{BASE}$和$Bert_{Large}$在所有任务上的性能都大大优于所有模型，分别比现有最高水准的模型提高了4.5%和7.0%的平均精度。注意，除了attention masking之外，$BERT_{BASE}$和OpenAI GPT在模型架构方面几乎是相同的。对于GLUE任务中最大的以及引用最广的任务MNLI,BERT在绝对精度上提高了4.6%，我们发现$Bert_{Lrage}$在所有任务中都明显优于BERT$_{BASE}$，尤其是那些训练数据很少的任务。

#### 4.2 SQuAD v1.1

斯坦福问答数据集(SQuAD v1.1)包含10万个问答对。该任务在给定问题和包含答案的文章下去预测这篇文章的答案文本片段。

对于问答任务，作者将问题和文章打包成序列做为输入，在微调阶段，仅引进开始向量$S \in \mathbb{R}^H$和 结束向量$E \in \mathbb{R}^H$。单词$i$是答案片段的开始单词的概率$P_i=\frac{e^{S \cdot T_i}}{\sum_{j}e^{S \cdot T_j}}$。计算答案片段的末尾单词同样可以用这个公式。从$i$到$j$的候选答案片段的得分定义为$S\cdot T_i + E\cdot T_j$,其中$j \geq i$。

Table 2展示了问答任务中的排名前几位的模型，因此，作者使用适度的数据增强，首先对TriviaQA进行微调，然后再对SQuAD进行微调。 BERT在集成度上比排名第一的系统高出+1.5f1，在单个系统上比排名第一的系统高出+1.3f1。

![Table 2](../../../../../../img/in-post/2020.04/16/Table 2.jpg)
Table 2: SQuAD 1.1 results. The BERT ensemble is 7x systems which use different pre-training checkpoints and fine-tuning seeds.

![Table 3](../../../../../../img/in-post/2020.04/16/Table 3.jpg)
Table 3: SQuAD 2.0 results. We exclude entries that use BERT as one of their components.

#### 4.3 SQuAD v2.0

SQuAD2.0任务扩展了SQuAD1.1问题的定义，允许在提供的段落中不存在短答案的可能性，从而使问题更加符合现实。

对于这个2.0任务，作者使用一个简单的方法来扩展SQuAD1.1中的Bert模型。作者将那些没有答案的问题视为有答案，且答案的开始和结束都在[CLS]标记处<font color="red">(这里理解不知道对不对)</font>，在预测时，无答案的片段得分计算为：$S_{null}=S\cdot C+E\cdot C$,对于最好的非空片段$S_{\hat{i},j}=max_{j\geq i}S\cdot T_i+E\cdot T_j$,当$S_{\hat{i},j}\gt S_{null}+\pi$,可以预测该问题是有答案的，其中，阈值$\pi$是在开发集中最大的F1值。

![Table 4](../../../../../../img/in-post/2020.04/16/Table 4.jpg)
Table 4: SWAG Dev and Test accuracies. †Human performance is measured with 100 samples, as reported in
the SWAG paper.

在SQuAD2.0任务中，BERT比之前最好的模型改进了5.1个F1分数。

#### 4.4 SWAG

$Bert_{Large}$的性能比基线ESIM+ELMo系统高出27.1%，OpenAI GPT高出8.3%。

###  <span id="p6">五、Ablation Studies</span>
作者对BERT的许多组件进行消除实验，以便更好地理解它们的重要性。
![Table 5](../../../../../../img/in-post/2020.04/16/Table 5.jpg)
Table 5: Ablation over the pre-training tasks using the BERTBASE architecture. “No NSP” is trained without the next sentence prediction task. “LTR & No NSP” is trained as a left-to-right LM without the next sentence prediction, like OpenAI GPT. “+ BiLSTM” adds a randomly initialized BiLSTM on top of the “LTR + No NSP” model during fine-tuning.

#### 5.1 Effect of Pre-training Tasks

通过使用与$BERT_{BASE}$完全相同的训练前数据、微调方案和超参数评估两个训练前目标，作者证明了BERT的深度双向性的重要性：

<strong>No NSP:</strong>双向性模型使用MLM(masked LM)进行训练，在NSP(next sentence prediction)任务中是没有部分的。

<strong>LTR & No NSP:</strong>left-context-only模型使用标准的自左向右的LM进行训练，而不用MLM。在微调时也应用了left-only约束，因为删除它会导致预训练/微调不匹配，从而降低下游性能。

在Table 5中可以看出，移除NSP会显著影响QNLI，MNLI和SQuAD1.1的性能。作者通过比较“No NSP”和“ LTR＆No NSP”来评估训练的双向性表示的影响。LTR模式在所有任务上的表现都比MLM模型差，在MRPC和SQuAD上的性能都有很大的下降。

对于SQuAD而言，显LTR模型在单词预测方面表现不佳，因为词级级隐藏状态没有右侧文本内容。为了强化LTR系统，作者随机地在顶部初始化BiLSTM。这确实改善了SQuAD的结果，但是仍然远不如那些预训练的双向模型。BiLSTM会影响GLURE任务的性能。

虽然可以训练单独的LTR和RTL模型，并像ELMo那样将每个单词表示为两个模型的连接。但是会有如下弊端：
* a)效率是单个双向模型1/2；
* b)对于QA这样的任务来说，这是不直观的，因为RTL模型无法对问题的答案进行限定；
* c)这比深度双向模型的功能要弱得多，因为它可以在每一层同时使用左上下文和右上下文。

#### 5.2 Effect of Model Size

在本节中探讨模型大小对微调任务精度的影响。作者训练了许多具有不同层数、隐藏单元和注意头的BERT模型，而在其他方面则使用与前面描述的相同的超参数和训练过程。

所选的GLUE任务的结果如Table 6所示。我们可以看到，更大的模型导致所有四个数据集的精确性得到了一定的提高，即使对于只有3600个词的训练例子的MRPC，也与预训练任务有很大的不同。
![Table 6](../../../../../../img/in-post/2020.04/16/Table 6.jpg)
Table 6: Ablation over BERT model size. #L = the number of layers; #H = hidden size; #A = number of attention heads. “LM (ppl)” is the masked LM perplexity of held-out training data.

#### 5.3 Feature-based Approach with BERT

到目前为止，所有的BERT结果都使用了微调方法，即在预先训练的模型中添加一个简单的分类层，并在下游任务中联合微调所有参数。然而，基于特征的方法从预训练模型中提取固定特征具有一定的优势。首先，并不是所有的任务都可以很容易地用一个Transformer-encoder架构来表示，因此需要添加一个特定于任务的模型架构。其次，预先计算一次昂贵的训练数据表示，然后在这种表示的基础上使用更便宜的模型运行许多实验，这有很大的计算优势。
![Table 7](../../../../../../img/in-post/2020.04/16/Table 7.jpg)
Table 7: CoNLL-2003 Named Entity Recognition results. Hyperparameters were selected using the Dev set. The reported Dev and Test scores are averaged over 5 random restarts using those hyperparameters.

为了消除这种微调方法，作者采用基于特征的方法，从一个或多个层中提取激活信息，而不需要微调BERT的任何参数。这些上下文嵌入用作在分类层之前随机初始化的两层768维BiLSTM的输入，结果见Table 7。性能最好的方法是将预先训练好的转换器的前四个隐藏层中的词表示连接起来，这只比微调整个模型落后0.3个f1。这表明BERT对于精细调整和基于特征的方法都是有效的。

###  <span id="p7">Conclusion</span>

近年来，基于语言模型的迁移学习(transfer)的实证研究表明，丰富的、无监督的预训练是许多语言理解系统的一个组成部分。特别是，这些结果使得即使是低资源任务也能从深度单向架构中获益。本文的主要贡献是将这些发现进一步推广到深层双向架构中，使相同的预训练模型能够成功地处理一系列NLP任务。
