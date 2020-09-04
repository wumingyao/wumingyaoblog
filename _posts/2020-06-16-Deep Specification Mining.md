---
layout:     post
title:      "《Deep Specification Mining》"
subtitle:   "深度规格挖掘论文阅读"
date:       2020-06-16
author:     "wumingyao"
header-img: "img/in-post/2020.06/16/bg.jpg"
tags: [深度规格挖掘]
categories: [论文笔记]
---
## 主要内容
* [ABSTRACT](#p1)
* [1 INTRODUCTION](#p2)
* [2 BACKGROUND](#p3)
* [3 PROPOSED APPROACH](#p4)
* [4 EMPIRICAL EVALUATION](#p5)
* [5 MINING FSA FOR DETECTING ANDROID MALICIOUS BEHAVIORS](#p6)
* [6 THREATS TO VALIDITY](#p7)
* [7 RELATED WORK](#p8)
* [8 CONCLUSION AND FUTURE WORK](#p9)

## 正文

###  <span id="p1">ABSTRACT</span>
Formal specifications are essential but usually unavailable in software systems. Furthermore, writing these specifications is costly and requires skills from developers. Recently, many automated techniques have been proposed to mine specifications in various formats including finite-state automaton (FSA). However, more works in specification mining are needed to further improve the accuracy of the inferred specifications.

In this work, we propose Deep Specification Miner (DSM), a new approach that performs deep learning for mining FSA-based specifications. Our proposed approach uses test case generation to generate a richer set of execution traces for training a Recurrent Neural Network Based Language Model (RNNLM). From these execution traces, we construct a Prefix Tree Acceptor (PTA) and use the learned RNNLM to extract many features. These features are subsequently utilized by clustering algorithms to merge similar automata states in the PTA for constructing a number of FSAs. Then, our approach performs a model selection heuristic to estimate F-measure of FSAs and returns the one with the highest estimated Fmeasure. We execute DSM to mine specifications of 11 target library classes. Our empirical analysis shows that DSM achieves an average F-measure of 71.97%, outperforming the best performing baseline by 28.22%. We also demonstrate the value of DSM in sandboxing Android apps.

摘要:

说明了形式化规范的重要性、难点以及前人的工作，在此基础上，作者提出了一种基于递归神经网络语言模型（RNNLM）挖掘FSA规范的新方法DSM。使用著名的测试用例生成方法Randoop来创建一组更丰富的执行跟踪来训练RNNLM。从一组采样的执行轨迹中构造一个前缀树接受器（PTA），并利用学习的RNNLM提取PTA状态的许多特征。然后，聚类算法利用这些特征合并相似的自动机状态，使用各种设置构造许多fsa。然后采用一种启发式的模型选择方法来选择最精确的FSA，并将其作为最终模型输出。作者运行所提出的方法来推断11个目标库类的规范。结果表明，DSM的平均F-测度为71.97%，比最优基线的平均F-测度高28.82%。此外，作者还提出了一种利用DSM挖掘的FSA来检测Android应用程序中恶意行为的技术。特别是，使用推断出的FSA作为行为模型来构造一个更全面的沙盒，该沙盒考虑敏感API方法的执行上下文。结果表明，该方法能使黄蜂的真阳性率提高15.69%，而假阳性率仅提高4.52%。

###  <span id="p2">1 INTRODUCTION</span>

Due to rapid evolution to meet demands of clients, software applications and libraries are often released without documentedspecifications. Even when formal specifications are available, they may become outdated as software systems quickly evolve  in a short period of time. Finally, writing formal specifications requires necessary skill and motivation from developers, as this is a costly and time consuming process . Furthermore, the lack of specifications negatively impacts the maintainability and reliability of systems. With no documented specifications, developers may find it difficult to comprehend a piece of code and software is more likely to have bugs due to mistaken assumptions. Furthermore, developers cannot utilize state-of-the-art bug finding and testing tools that need formal specifications as an input.

作者分析了现状，当前软件开发都是以快速开发为主，所以在开发之前没有正式的规格说明文档，另外
编写规格文档需要开发人员具备必要的技能，这个过程是耗时耗力的。

Recently, many automated approaches have been proposed to help developers reduce the cost of manually drafting formal specifications. In this work, we focus on the family of specification mining algorithms that infer finite-state automaton (FSA) based specifications from execution traces. Krka et al.  and many other researchers have proposed various FSA-mining approaches that have improved the quality of inferred FSA models as compared to prior solutions. Nevertheless, the quality of mined specifications is not perfect yet, and more works need to be done to make specification mining better. In fact, FSA based specification miners still suffer from many issues. For instance, if methods in input execution traces frequently occur in a particular order or the amount of input traces is too small, FSAs inferred by k-tails and many other algorithms are likely to return FSAs that are not generalized and overfitted to the input execution traces.

作者调研了自动化挖掘规范的相关工作，总结了前人工作的优缺点。目前的规范挖掘算法从执行轨迹中推断出基于有限状态自动机（FSA）的规范。
与先前的解决方案相比，这些方法提高了推断的FSA模型的质量。然而，目前采出的规格的质量还不完善，需要做更多的工作，使规格挖掘更好。

基于FSA的规范挖掘者仍然面临许多问题：
例如，如果输入执行跟踪中的方法经常以特定的顺序出现，或者输入跟踪的数量太少，则由k-tails和许多其他算法推断的fsa可能会返回未被泛化和过度拟合到输入执行跟踪的fsa。

To mine more accurate FSA models, we propose a new specification mining algorithm that performs deep learning on execution traces. We name our approach DSM which stands for Deep Specification Miner. Our approach takes as input a target library class C and employs an automated test case generation tool to generate thousands of test cases. The goal of this test case generation process is to capture a richer set of valid sequences of invoked methods of C. Next, we perform deep learning on execution traces of generated test cases to train a Recurrent Neural Network Language Model (RNNLM). After this step, we construct a Prefix Tree Acceptor (PTA) from the execution traces and leverage the learned language model to extract a number of interesting features from PTA’s nodes. These features are then input to clustering algorithms for merging similar states (i.e., PTA’s nodes). The output of an application of a clustering algorithm is a simpler and more generalized FSA that reflects the training execution traces. Finally, our approach predicts the accuracy of constructed FSAs (generated by different clustering algorithms considering different settings) and outputs the one with highest predicted value of F-measure.

作者提出了一种新的规格文档挖掘算法用于挖掘更为精确的FSA模型，这个算法主要是对执行轨迹进行深度学习。
作者利用自动生成测试用例的工具生成数千个测试用例，并将这些测试用例做为训练样本对执行轨迹进行深度学习，
以训练递归神经网络语言模型（RNNLM）。随后，作者从执行轨迹中构造一个前缀树接受器（PTA），
并利用所学习的语言模型从PTA的节点中提取特征，最后，通过聚类生成FSA。

We evaluate our proposed approach for 11 target library classes which were used before to evaluate many prior work . For each of the input class, we first run Randoop to generate thousands of test cases. Then, we use execution traces generated by running these test cases to infer FSAs. Our experiments show that DSM achieves an average F-measure of 71.97%. Compared to other existing specification mining algorithms, our approach outperforms all baselines that construct FSAs from execution traces  by at least 28.22%. Some of the baselines first use Daikon to learn invariants that are then used to infer a better FSA. Our approach does not use Daikon invariants in the inference of FSAs. Excluding baselines that use Daikon invariants, our approach can outperform the remaining best performing miner by 33.24% in terms of average F-measure.

作者针对11个目标库类评估了所提出的方法，实验表明，DSM的平均F-测度为71.97%。与其他现有的规范挖掘算法相比，作者的方法的性能比所有从执行跟踪构造fsa的基线至少高出28.22%。

Additionally, we assess the applicability of FSAs mined by DSM in detecting malicious behaviors in Android apps. We propose a technique that leverages a FSA output by DSM mining algorithm as a behavior model to construct an Android sandbox. Our technique outputs a comprehensive sandbox that considers execution context of sensitive API methods to better protect app users. Our comparative evaluation finds that our technique can increase the True Positive Rate of Boxmate, a state-of-the-art sandbox mining approach, by 15.69%, while only increasing False Positive Rate by 4.52%. Replacing DSM with the best performing applicable baseline results in a sandbox that can achieve a similar True Positive Rate (as DSM) but substantially worse False Positive Rate (i.e., False Positive Rate increases by close to 10%). The results indicate it is promising to employ FSAs mined by DSM to create more effective Android sandboxes.

此外，作者还评估了DSM挖掘的fsa在Android应用程序恶意行为检测中的适用性。研究结果表明，利用DSM挖掘的FSA来创建更有效的Android沙盒是有希望的。

The contributions of our work are highlighted below:     
(1)	We propose DSM (Deep Specification Miner), a new specification mining algorithm that utilizes test case generation, deep learning, clustering, and model selection strategy to infer FSA based specifications. To the best of our knowledge, we are the first to use deep learning for mining specifications.      
(2)	We evaluate the effectiveness of DSM on 11 different target library classes. Our results show that our approach outperforms the best baseline by a substantial margin in terms of average F-measure.                  
(3)	We propose a technique that employs a FSA inferred by DSM to construct a more comprehensive sandbox that considers execution context of sensitive API methods. Our evaluation shows that our proposed technique can outperform several baselines by a substantial margin in terms of either True Positive Rate or False Positive Rate.      

本研究的贡献：        
(1)提出了一种新的规范挖掘算法DSM      
(2)在11个不同的目标库类上评估了DSM的有效性。      
(3)提出一种技术，利用DSM推断出的FSA构造一个更全面的沙盒，考虑敏感API方法的执行上下文。      

###  <span id="p2">BACKGROUND</span>

本章介绍了两种语言模型，统计语言模型和基于语言的循环神经网络模型。

#### Statistical Language Model

A statistical language model is an oracle that can foresee how likely a sentence $s=w_1,w_2,\cdots,w_n$ 
to occur in a language. In a nutshell, a statistical language model considers a sequence s to be a list of words $w_1,w_2,\cdots,w_n$ and assigns probability to s by computing joint probability of words: $P(w_1,\cdots,w_n) =\prod_{i=0}^{n-1}P(w_i | w_1,\cdots,w_{i−1})$. As it is challenging to
compute conditional probability $P(w_i | w_1,\cdots,w_{i−1})$, each different language model has its own assumption to approximate the calculation. N-grams model, a popular family of language models, approximates in a way that a word wk conditionally depends only on its previous N words (i.e., $w_{k−N+1},\cdots,w_{k−1})$. For example, unigram model simply estimates $P(w_i | w_1,\cdots,w_{i−1})$ as $P(w_i)$, bigram model approximates $P(w_i | w_1,\cdots,w_{i−1})$ as $P(w_i | w_{i−1})$, etc. In this work, we utilize the ability of language models to compute
$P(w_i | w_1,\cdots,w_{i−1})$ for estimating features of automaton states. We consider every method invocation as a word and an execution trace of an object as a sentence (i.e., sequence of method invocations). Given a sequence of previously invoked methods, we use a language model to output the probability of a method to be invoked next.

介绍了统计语言模型，统计语言模型是用来预测某个句子出现的概率的模型。

#### Recurrent Neural Network Based Language Model
Recently, a family of language models that make use of neural networks is shown to be more effective than n-grams. These models are referred to as neural network based language models (NNLM). If a NNLM has many hidden layers, we refer to the model as a deep neural network language model or deep language model for short. Among these deep language models, Recurrent Neural Network Based Language Model (RNNLM) is well-known with its ability to use internal memories to handle sequences of words with arbitrary lengths. The underlying network architecture of a RNNLM is a Recurrent Neural Network (RNN) that stores information of input word sequences in its hidden layers. Figure 1 demonstrates how a RNN operates given the sequence <START>, STN, NT, HMTF, <END>. In the figure, a RNN is unrolled to become four connected networks, each of which is processing one input method at a time step. Initially, all states in the hidden layer are assigned to zeros. At time $t_k$, a method $m_k$ is represented as an one-hot vector $i_k$ by the input layer. Next, the hidden layer updates its states by using the vector $i_k$ and the states previously computed at time $t_{k−1}$. Then, the output layer estimates a probability vector ok across all methods for them to appear in the next time step $t_{k+1}$. This process is repeated at subsequent time steps until the last method in the sequence is handled.

介绍了基于语言的循环神经网络模型，它是一个深度学习模型，可以处理任意长度的单词序列。

###  <span id="p3">PROPOSED APPROACH</span>

Figure 2 shows the overall framework of our proposed approach. In our framework, there are three major processes: test case generation and traces collection, Recurrent Neural Network Based Language Model (RNNLM) learning, and automata construction. Our approach takes as input a target class and signatures of methods. Then, DSM runs Randoop  to generate a substantial number of test cases for the input target class. Then, we record the execution of these test cases, and retain traces of invocations of methods of the input target class as the training dataset. Next, our approach performs deep learning on the collected traces to infer a RNNLM that is capable of predicting the next likely method to be executed given a sequence of previously called methods. We choose RNNLM over traditional probabilistic language models since past studies show its superiority.

作者概括了所提议的方法的总体框架。在这个框架中，主要有三个过程：测试用例生成和跟踪收集、基于递归神经网络的语言模型（RNNLM）学习和自动机构建。
该模型输入目标类和方法签名，然后，DSM运行Randoop为输入目标类生成大量测试用例。然后，记录这些测试用例的执行，并将对输入目标类的方法的调用的跟踪保留为训练数据集。
接下来，对收集到的跟踪执行深入学习，用于训练RNNLM，该RNNLM能够预测给定一系列先前调用的方法将要执行的下一个可能的方法。


![Figure 2](../../../../../../img/in-post/2020.06/16/Figure 2.jpg)
Figure 2: DSM’s Overall Framework

Subsequently, we employ a heuristic to select a subset of traces that best represents the whole training dataset. From these traces, we construct a Prefix Tree Acceptor (PTA); we refer to each PTA’s node as an automaton state. We select the subset of traces in order to optimize the performance when constructing PTA, but still maintaining accuracy of inferred FSAs. Utilizing the inferred RNNLM, we extract a number of features from automaton states, and input the feature values to a number of clustering algorithms (i.e., k-means  and hierarchical clustering  ) considering different settings (e.g., different number of clusters). The output of a clustering algorithm are clusters of similar automaton states. We use these clusters to create a new FSA by merging states that belong to the same cluster. Every application of a clustering algorithm with a particular setting results in a different FSA. We propose a model selection strategy to heuristically select the most accurate model by predicting values of Precision, Recall, and F-measure. Finally, we output the FSA with highest predicted F-measure.

根据这些轨迹，作者构造一个前缀树接受器(PTA)。
将每个PTA的节点称为一个自动机状态。利用推断出的RNNLM，从自动机状态中提取出若干特征，并将这些特征值输入到考虑不同设置（如不同簇数）的若干聚类算法（即k-均值和层次聚类）中。聚类算法的输出是相似自动机状态的聚类。
作者使用这些集群通过合并属于同一集群的状态来创建新的FSA。具有特定设置的聚类算法的每个应用都会导致不同的FSA。

#### 3.1 Test Case Generation and Trace Collection

This process plays an important role to our approach as it decides the quality of RNNLM inferred by the deep learning process. Previous research works in specification mining collect traces from the execution of a program given unit test cases or inputs manually created by researchers. In this work, we utilize deep learning for mining specification. Deep learning requires a substantially large and rich amount of data. The more training inputs, the more patterns the resultant RNNLM can capture. In general, it is difficult to follow previous works to collect a rich enough set of execution traces for an arbitrary target library class. Firstly, it is challenging to look for all projects that use the target library class, especially for classes from new or unreleased libraries. Secondly, existing unit test cases or manually created inputs may not cover many of the possible execution scenarios of methods in a target class.

作者利用深度学习来挖掘规范。深度学习需要大量丰富的数据。训练输入越多，生成的RNNLM可以捕获的模式越多。

We address the above issues by following Dallmeier to generate as many test cases as possible for mining specifications, and collect the execution traces of these test cases for subsequent steps. Recently, many test case generation tools have been proposed such as Randoop etc. Among the state-ofthe-art test case generation tools, we choose Randoop because it is widely used and lightweight. Furthermore, Randoop is well maintained and frequently updated with new versions. As future work, we plan to integrate many other test case generation methods into our approach.

Randoop是一款测试用例生成工具。作者选择Randoop是因为它被广泛使用和轻量级。此外，Randoop维护良好，并经常更新新版本。

Randoop generates a large number of test cases, which is proportional to the time limit of its execution. In order to improve the coverage of possible sequences of methods under test, we provide class-specific literals aside from default ones to Randoop. For example, for java.net.Socket, we create string and integer literals which are addresses of hosts (e.g., “localhost”, “127.0.0.1”, etc.) and listening ports (e.g., 8888, etc.). Furthermore, we create driver classes that contain static methods that invoke constructors of the target class to initialize new objects. That helps speed up Randoop to create new objects without spending time to search for appropriate input values for constructors.

作者创建包含静态方法的驱动程序类等方法，调用目标类的构造函数来初始化新对象。这有助于加快Randoop创建新对象的速度，而无需花时间搜索构造函数的适当输入值。

#### 3.2 Learning RNNLM for Specification Mining
##### 3.2.1 Construction of Training Method Sequences

Our set of collected execution traces is a series of method sequences. Each of these sequences starts and ends with two special symbols: <START> and <END>, respectively. These symbols are used for separating two different sequences. We gather all sequences together to create data for training Recurrent Neural Networks. Furthermore, we limit the maximum frequency of a method sequence MAX_SEQ_FREQ to 10 to prevent imbalanced data issue where a sequence appears much more frequently than the other ones.
 
作者介绍了构建训练样本，收集的一组执行跟踪是一系列方法序列。每个序列都以两个特殊符号开始和结束：分别是<START>和<END>。这些符号用于分隔两个不同的序列。我们将所有序列集合在一起，为训练递归神经网络创建数据。此外，我们将方法序列MAX_SEQ_FREQ的最大频率限制为10，以防止序列比其他序列更频繁出现的不平衡数据问题。

##### 3.2.2 Model Training
We perform deep learning on the training data to learn a Recurrent Neural Network Based Language Model (RNNLM) for every target library class. By default, we use Long Short-Term Memory (LSTM) network , one of the stateof-the-art RNNs, as the underlying architecture of the RNNLM. Compared to the standard RNN architecture, LSTM is better in learning long-term dependencies. Furthermore, LSTM is scalable for long sequences.

作者介绍了模型的训练方法。作者对训练数据进行深度学习，为每个目标库类学习一个基于递归神经网络的语言模型（RNNLM）。

#### 3.3 Automata Construction

In this processing step, our approach takes as input the set of training execution traces and the inferred RNNLM (see Section 3.2). The output of this step is a FSA that best captures the specification of the corresponding target class. The construction of FSA undergoes several substeps: trace sampling, feature extraction, clustering, and model selection.

自动化构建这个步骤的输出是一个FSA，它最好地捕获对应目标类的规范。FSA的构建经历了跟踪采样、特征提取、聚类和模型选择几个步骤。

At first, we use a heuristic to select a subset of method sequences that represents all training execution traces. The feature extraction and clustering steps use these selected traces, instead of all traces, to reduce computation cost. We construct a Prefix Tree Acceptor (PTA) from the selected traces and extract features for every PTA nodes using the inferred RNNLM. We refer to each PTA node as an automaton state. Figure 3 shows an excerpt of an example PTA constructed from sequences of invocations of methods from java.security.Signature. Our goal is to find similar automaton states and group them into one cluster. In the clustering substep, we run a number of clustering algorithms on PTA nodes with various settings to create many different FSAs. Finally, in the model selection substep, we follow a heuristic to predict the F-measure (see Section 4.2.1) of constructed FSAs and output the one with highest predicted F-measure. The full set of traces is used in this model selection step. In the following paragraphs, we describe details of each substep in this processing step:

作者详细介绍了自动化构建的方法步骤。首先，作者使用启发式方法来选择表示所有训练执行轨迹的方法序列子集。特征提取和聚类步骤使用这些选定的跟踪，以降低计算成本。从所选的记录道中构造一个前缀树接收器（PTA），并使用推断的RNNLM为每个PTA节点提取特征。作者的目标是找到相似的自动机状态并将它们组合成一个集群。在聚类子步骤中，我们在PTA节点上运行了许多具有不同设置的聚类算法，以创建许多不同的fsa。最后，在模型选择子步骤中，作者采用启发式方法预测构造的fsa的F-测度，并输出预测F-测度得分最高的一个。

Trace Sampling: Our training data contains a large number of sequences. Thus, it is expensive to use all of them for constructing FSAs. Therefore, the goal of trace sampling is to create a smaller subset that is likely to represent the whole set of all traces reasonably well. We propose a heuristic to find a subset of traces that covers all cooccurrence pairs  of methods in all training traces.

作者介绍了跟踪采样的过程。

Feature Extraction: From method sequences of the sampled execution traces, we construct a Prefix Tree Acceptor (PTA). A PTA is a tree-like deterministic finite automaton (DFA) created by putting all the prefixes of sequences as states, and a PTA only accepts the sequences that it is built from. The final states of our constructed PTAs are the ones have incoming edges with <END> labels (see Section 3.2). Figure 3 shows an example of a Prefix Tree Acceptor (PTA). Table 1 shows information of the extracted features. For each state S of a PTA, we are particularly interested in two types of features:

(1)	Type I: This type of features captures information of previously invoked methods before the state S is reached. The values of type I features for state S is the occurrences of methods on the path between the starting state (i.e., the root of the PTA) and S. For example, according to Figure 3, the values of Type I features corresponding to node S3 are: 
$F_{\langle START\rangle}=F_{\langle init\rangle}=F_{initVerify}=1$ 
and $F_{update} = F_{verify} = F_{\langle END\rangle} = 0$.

(2)	Type II: This type of features captures the likely methods to be immediately called after a state is reached. Values of these features are computed by the inferred RNNLM in the deep learning step (see Section 3.2). For example, at node S3 in Figure 3, initVerify and <END> have higher probabilities than the other methods to be called afterward. Examples of type II features and their values for node S3 output by a RNNLM are as follows: 
$P_{initVerify} = P_{\langle END\rangle} = 0.4$ and $P_{\langle START\rangle} = P_{\langle init\rangle} = P_{verify} = P_{update} = 0.15$.

作者主要提取两种类型的特征。 类型I：这类特性在到达状态S之前捕获以前调用的方法的信息。状态S的类型I特征值是在起始状态（即PTA的根）和S之间的路径上出现的方法。
类型II：这种类型的特性捕获在到达状态后立即调用的可能方法。

Clustering: We run k-means  and hierarchical clustering algorithms on the PTA’s states with their extracted features. Our goal is to create a simpler and more generalized automaton that captures specifications of a target library class. Since both k-means and hierarchical clustering require the predefined inputC for number of clusters, we try with many values ofC from 2 to MAX_CLUSTER (refer to Section 4.2.2) to search for the best FSA. Overall, the execution of clustering algorithms results in 2 × (MAX_CLUSTER − 1) FSAs.

作者介绍了聚类过程。作者使用k-均值和层次聚类算法对PTA的状态及其提取的特征进行聚类。我们的目标是创建一个更简单、更通用的自动机，它捕获目标库类的规范。

Model Selection: We propose a heuristic to select the best FSA among the ones output by the clustering algorithms. Algorithm 2 describes our strategy to predict precision of an automaton M given the set of all traces Data (see Section 3.1).

作者这部分说明了模型的选择。作者提出了一个启发式的方法来从聚类算法的输出中选择最佳的FSA。

###  <span id="p4"> EMPIRICAL EVALUATION</span>
#### 4.1 Dataset
##### 4.1.1 Target Library Classes
In our experiments, we select 11 target library classes as the benchmark to evaluate the effectiveness of our proposed approach. These library classes were investigated by previous research works in specification mining. Table 2 shows further details of the selected library classes including information of collected execution traces. Among these library classes, 9 out of 11 are from Java Development Kit (JDK); the other two library classes are DataStructure.StackAr (from Daikon project) and NumberFormatStringTokenizer (from Apache Xalan). For every library class, we consider methods that were analyzed by Krka et al.

作者选取了11个目标库类作为基准来评估所提出的方法的有效性。在这些库类中，11个库类中有9个来自Java开发工具包（JDK）；另外两个库类是数据结构和数字格式字符串标记器。

##### 4.1.2 Ground Truth Models
We utilize ground truth models created by Krka et al. Among the investigated library classes, we refine ground truth models of five Java’s Collection based library classes (i.e., ArrayList, LinkedList, HashMap, HashSet, and Hashtable) to capture “empty” and “non-empty” Table 2: Target Library Classes. “#M” represents the number of class methods that are analyzed, “#Generated Test Cases” is the number of test cases generated by Randoop, “#Recorded Method Calls” is the number of recorded method calls in the execution traces, “NFST” stands for NumberFormatStringTokenizer.

作者利用Krka等人建立的基线模型。改进了五个基于Java集合的库类的模型，以捕获“空”和“非空”的目标库类。

#### 4.2 Experimental Settings
##### 4.2.1 Evaluation Metrics
We follow Lo and Khoo’s method  to measure precision and recall for assessing the effectiveness of our proposed approach. Lo and Khoo’s method has been widely adopted by many prior specification mining works . Their proposed approach takes as input a ground truth and an inferred FSA. Next, it generates sentences (i.e., traces) from the two FSAs to compute their similarities. Precision of an inferred FSA is the percentage of sentences accepted by its corresponding ground truth model among the ones that are generated by that FSA. Recall of an inferred FSA is the percentage of sentences accepted by itself among the ones that are generated by the corresponding ground truth model. In a nutshell, precision reflects the percentage of sentences produced by an inferred model that are correct, while recall reflects the percentage of correct sentences that an inferred model can produce. We use F-measure, which is the harmonic mean of precision and recall, as a summary metric to evaluate specification mining algorithms. F-measure is defined as follows:
$$F-Measure=2\times\frac{Precision\times Recall}{Precision + Recall}\tag{1}$$

作者介绍了评价指标F-Measure,用于评估方法的有效性。
精确性反映了推断模型生成的正确句子的百分比，而召回率反映了推断模型生成的正确句子的百分比。作者使用精确性和召回率的调和平均值F-测度作为总结度量来评估规范挖掘算法。

To accurately compute precision, recall and F-measure, sentences generated from a FSA must thoroughly cover its states and transitions. To achieve that goal, we set the maximum number of generated sentences to 10,000 with maximal length of 50, and minimum coverage of each transition equals to 20. Similar strategies were adopted in prior works.
为了准确计算精确性、召回率和F-测度，FSA生成的句子必须完全覆盖其状态和转换。

##### 4.2.2 Experimental Configurations & Environments
Randoop Configuration. In test case generation step, for each target class, we repeatedly execute Randoop (version 3.1.2) with a time limit of 5 minute with 20 different initial seeds. We set the time limit to 5 minutes to make sure subsequent collected execution traces are not too long as well as not too short. We repeat execution of Randoop 20 times to maximize the coverage of possible sequences of program methods in Randoop generated test cases. Furthermore, we turn off Randoop’s option of generating errorrevealing test cases (i.e., --no-error-revealing-tests is set to true) as executions of these test cases are usually interrupted by exceptions or errors, which results in incomplete method sequences for subsequent deep learning process. We find that with this setup the generated traces cover 100% of target API methods; also, on average, 96.97% and 98.18% of edges and states in each target class’ ground-truth model are covered.

作者介绍了Randoop的配置。在测试用例生成步骤中，对于每个目标类，作者使用20个不同的初始种子以5分钟的时间限制重复执行Randoop。

##### 4.2.3 Baselines
In the experiments, we compare the effectiveness of DSM with many previous specification mining works . Krka et al. propose a number of algorithms that analyze execution traces to infer FSAs [24]. These algorithms are k-tails, CONTRACTOR++, SEKT, and TEMI. CONTRACTOR++, TEMI, and SEKT infer models leveraging invariants learned using Daikon. On the other hand, k-tails construct models only from ordering of methods in execution traces. Despite the fact that DSM is not processing likely invariants, we include CONTRACTOR++, SEKT, and TEMI as baselines to compare the applicability of deep learning and likely invariant inference in specification mining. For k-tails and SEKT, we choose k ∈ {1, 2} following Krka et al. [24] and Le et al. [26]’s configurations. In total, we have six different baselines: Traditional 1-tails, Traditional 2-tails,CONTRACTOR++, SEKT 1-tails. SEKT 2-tails, and TEMI.
 
作者将自己的模型与其他基线做了实验进行对比。

###  <span id="p5"> MINING FSA FOR DETECTING ANDROID MALICIOUS BEHAVIORS</span>

Nowadays, Android is the most popular mobile platform with millions of apps and supported devices. As the matter of fact, Android users easily become targets of attackers. Recently, several approaches have been proposed to protect users from potential threats of malware. Among state-of-the-art approaches, Jamrozik et al. propose Boxmate that mines rules to construct Android sandboxes by exploring behaviors of target benign apps. The key idea of Boxmate is it prevents a program to change its behaviour; it can prevent hidden attacks, backdoors, and exploited vulnerabilities from compromising the security of an Android app. Boxmate works on two phases: monitoring and deployment. In the monitoring phase, Boxmate employs a test case generation tool, named Droidmate, to create a rich set of GUI test cases. During the execution of these test cases, Boxmate records invocations of sensitive API methods (e.g., methods that access cameras, locations, etc.), and use them to create sandbox rules. The rules specify what sensitive API methods are allowed to be invoked during deployment. During deployment, when an app accesses a sensitive API method that is not recorded in the above rules, the sandbox immediately intercepts that operation and raises warning messages to users about the suspicious activity.

![Figure 5](../../../../../../img/in-post/2020.06/16/Figure 5.jpg)
Figure 5: Malware detection framework leveraging behavior models inferred by DSM

图5展示了恶意软件检测系统的拟议框架。该框架有两个阶段：

Monitoring Phase: This phase accepts as input a benign version of the target Android app. We first leverage GUI test case generation tools (i.e., Monkey [1], GUIRipper [2], PUMA [18], Droidmate [22], and Droidbot [30]) to create a diverse set of test cases. Next, we execute the input app with generated test cases and monitor API methods called. In particular, every time the app invokes a sensitive API method X, we select a sequence ofW previously executed API methods before X and include them to the training traces. Then, we employ DSM’s mining algorithm on the gathered traces to construct a FSA based behavior model BM. The constructed model reflects behaviors of the app when calling sensitive API methods. Subsequently, we employ BM to guide an automaton based sandbox in the deployment phase for malware detection.

这部分介绍了该框架的第一个阶段。第一个阶段是监控阶段，此阶段接受目标Android应用程序的良性版本作为输入。

Deployment Phase: In this phase, our framework leverages the inferred model BM to build an automaton based sandbox. The sandbox is used to govern and control execution of an Android app. Every time an app invokes a sensitive API X, the sandbox selects the sequence of W previously executed methods before X, and input them to the behavior model BM to classify the invocation of X as malicious or benign. If execution of X is predicted as malicious by model BM, the sandbox informs users by raising warning messages about suspicious activities. Otherwise, the sandbox allows the app to continue its executions without notifying users.

这部分介绍了该框架的第二个阶段。第二个阶段是部署阶段，在这个阶段在这个阶段，框架利用推断出的模型BM来构建一个基于自动机的沙盒。沙盒用于管理和控制Android应用程序的执行。

We evaluate our proposed malware detection framework using a dataset of 102 pairs of Android apps that were originally collected by Li et al. Each pair of apps contains one benign app and its corresponding malicious version. The malicious apps are created by injecting malicious code to their corresponding unpacked benign apps [29]. All these apps are real apps that are released to various app markets. Recently, Bao et al. [4] used the above 102 pairs to assess the effectiveness of Boxmate’s mined rules with 5 different test case generation tools (i.e., Monkey [1], GUIRipper [2], PUMA [18], Droidmate [22], and Droidbot [30]). In our evaluation, we utilize execution traces of the 102 Android app pairs collected by Bao et al. [4] . We setW (i.e., number of selected methods before an invoked sensitive API method) to 3, and employ these traces to infer several behavior models by using Boxmate, DSM as well as k-tails (k = 1). We include k-tails (k = 1) since according to Table 4 this is the best baseline mining algorithm that infers FSAs from raw traces of API invocations. We let the comparison between DSM and invariant based miners (i.e., CONTRACTOR++ and TEMI) for future work as Daikon is currently unable to mine invariants for Android apps. Next, we evaluate the effectiveness of inferred behavior models in detecting malware using the following evaluation metrics:

<strong>True Positive Rate(TPR)</strong>
$$TPR=\frac{TP}{TP+FN}\tag{2}$$

<strong>False Positive Rate(TPR)</strong>
$$FPR=\frac{FP}{FP+TN}\tag{3}$$

作者使用TPR和FPR两个评估指标评估推断的行为模型在检测恶意软件方面的有效性。

###  <span id="p6">THREATS TO VALIDITY</span>
Threats to internal validity. We have carefully checked our implementation, but there are errors that we did not notice. There are also potential threats related to correctness of ground truth models created by Krka et al. [24] that we used. To mitigate this threat, we have compared their models against execution traces collected from Randoop generated test cases as well as textual documentations published by library class writers (e.g., Javadocs). We revised the ground truth models accordingly.

Another threat to validity is related to parameter values of target API methods. We use traces collected by Bao et al. [4] which exclude all parameter values. This is different from Jamrozik et al.’s work [21] that excludes most (but not all) parameter values. We decide to exclude all parameter values since all specification mining algorithms considered in this paper (including DSM) produce FSAs that have no constraints on values of parameters. As future work, we plan to extend DSM to generate models that include constraints on parameter values.

作者验证了对内部有效性，并且相应地调整了模型和相应的参数值。

Threats to External Validity. These threats correspond to the generalizability of our empirical findings. In this work, we have analyzed 11 different library classes. This is larger than the number of target classes used to evaluate many prior studies, e.g., [24, 33, 34]. As future works, we plan to reduce this threat by analyzing more library classes to infer their automaton based specifications.

作者验证了对外部有效性的威胁。这些威胁与经验发现的普遍性相一致。在这项工作中，我们分析了11个不同的类库。这比用于评估许多先前研究的目标类别的数量要多。

Threats to Construct Validity. These threats correspond to the usage of evaluation metrics. We have followed Lo and Khoo’s approach that uses precision, recall, and F-measure to measure the accuracy of automata output by a specification mining algorithm against ground truth models [32]. Furthermore, Lo and Khoo’s approach is well known and has been adopted by many previous research works in specification mining e.g., [5, 6, 8, 13, 24, 27, 31, 33]. Additionally, True Positive Rate and False Positive Rate are wellknown metrics and widely adopted by state-of-the-art approaches in Android malware detection (e.g., [3, 50]).

威胁建构有效性。这些威胁对应于评估指标的使用。作者遵循了Lo和Khoo的方法，使用精确性、召回率和F-测度，通过针对基本真值模型的规范挖掘算法来测量自动机输出的准确性。

###  <span id="p7">RELATED WORK</span>
Mining Specifications. Aside from the state-of-the-art baselines considered in Section 4, there are other related works that mine FSA-based specifications from execution traces. Lo et al. propose SMArTIC that mines a FSA from a set of execution traces [13] using a variant of k-tails algorithm that constructs a probabilistic FSA. Mariani et al. propose k-behavior [36] that constructs an automaton by analyzing one single trace at a time. Walkinshaw and Bogdanov propose an approach that allows users to input temporal properties to support a specification miner to construct a FSA from execution traces [45]. Lo et al. further extend Walkinshaw and Bogdanov’s work to automatically infer temporal properties from execution traces, and use these properties to automatically support model inference process of a specification miner [33]. Synoptic infers three kinds of temporal invariants from execution traces and uses them to generate a concise FSA [6]. SpecForge [26] is a meta-approach that analyzes FSAs inferred by other specification miners and combine them together to create a more accurate FSA. None of the above mentioned approaches employs deep learning.

Deep Learning for Software Engineering Tasks. Recently, deep learning methods are proposed to learn representations of data with multiple levels of abstraction [28]. Researchers have been utilizing deep learning to solve challenging tasks in software engineering [17, 25, 47–49]. For example, Gu et al. propose DeepAPI that takes as input queries in natural languages and outputs sequences of API methods that developers should follow [17]. In the nutshell, DeepAPI replies on a RNN based model that can translate a sentence in one language to a new sentence in another language. Different from DeepAPI, DSM takes as input method sequences of an API or library and outputs a finite-state automaton that represents behaviors of that API or library. Prior to our work, deep learning models have not been employed to effectively mine specifications.

Language Models for Software Engineering Tasks. Statistical language models have been utilized for many software engineering tasks. For example, Hindle et al. employ n-gram model on code tokens to demonstrate a high degree of local repetitiveness in source code corpora and leverage it to improve Eclipse’s code completion engine [19]. Several other works have extended Hindle et al. work to build more powerful code completion engines; for example, Raychev et al. leverage n-gram and recurrent neural network language model to recommend likely sequences of method calls to a program with holes [41], while Nguyen et al. leverage Hidden Markov Model to learn API usages from Android app bytecode for recommending APIs [40]. Beyond code completion, Wang et al. use n-gram model to detect bugs by identifying low probability token sequences [46]. Our work uses language model for a different task.

这部分作者介绍了相关工作，包括规格文档挖掘、对于软件工程任务的深度学习以及对于软件工程的语言模型的构建等工作。

###  <span id="p8">CONCLUSION AND FUTURE WORK</span>

Formal specifications are helpful for many software processes. In this work, we propose DSM, a new approach that employs Recurrent Neural Network Based Language Model (RNNLM) for mining FSA-based specifications. We apply Randoop, a well-known test cases generation approach, to create a richer set of execution traces for training RNNLM. From a set of sampled execution traces, we construct a Prefix Tree Acceptor (PTA) and extract many features of PTA’s states using the learned RNNLM. These features are then utilized by clustering algorithms to merge similar automata states to construct many FSAs using various settings. Then, we employ a model selection heuristic to select the FSA that is estimated to be the most accurate and output it as the final model. We run our proposed approach to infer specifications of 11 target library classes. Our results show that DSM achieves an average F-measure of 71.97%, outperforming the best performing baseline by 28.82%. Additionally, we propose a technique that employs a FSA mined by DSM to detect malicious behaviors in an Android app. In particular, our technique uses the inferred FSA as a behavior model to construct a more comprehensive sandbox that considers execution context of sensitive API methods. Our evaluation shows that the proposed technique can improve the True Positive Rate of Boxmate by 15.69% while only increasing the False Positive Rate by 4.52%.

As future work, we plan to improve DSM’s effectiveness further by integrating information of likely invariants into our deep learning based framework. We also plan to employ EvoSuite [16] and many other test case generation tools to generate an even more comprehensive set of training traces to improve the effectiveness of DSM. Furthermore, we plan to improve DSM by considering many more clustering algorithms aside from k-means and hierarchical clustering, especially the ones that require no inputs for the number of clusters (e.g., DBSCAN [15], etc.). Finally, we plan to evaluate DSM with more classes and libraries in order to reduce threats to external validity.

作者对本研究进行了总结。本文提出了一种基于递归神经网络语言模型（RNNLM）挖掘FSA规范的新方法DSM。使用著名的测试用例生成方法Randoop来创建一组更丰富的执行跟踪来训练RNNLM。从一组采样的执行轨迹中构造一个前缀树接受器（PTA），并利用学习的RNNLM提取PTA状态的许多特征。然后，聚类算法利用这些特征合并相似的自动机状态，使用各种设置构造许多fsa。然后采用一种启发式的模型选择方法来选择最精确的FSA，并将其作为最终模型输出。作者运行所提出的方法来推断11个目标库类的规范。结果表明，DSM的平均F-测度为71.97%，比最优基线的平均F-测度高28.82%。此外，作者还提出了一种利用DSM挖掘的FSA来检测Android应用程序中恶意行为的技术。特别是，使用推断出的FSA作为行为模型来构造一个更全面的沙盒，该沙盒考虑敏感API方法的执行上下文。结果表明，该方法能使黄蜂的真阳性率提高15.69%，而假阳性率仅提高4.52%。

作为未来的工作，作者计划通过将可能不变量的信息集成到基于深度学习的框架中，进一步提高DSM的有效性。作者还计划使用其他测试用例生成工具来生成更全面的训练跟踪集，以提高DSM的有效性。此外，除了k-均值和层次聚类之外，还计划通过考虑更多的聚类算法来改进DSM，特别是那些不需要输入聚类数量的算法。最后作者计划使用更多的类和库来评估DSM，以减少对外部有效性的威胁。
