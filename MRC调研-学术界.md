MRC调研——学术界

## 1.任务

### 1.1 背景

 机器阅读理解（Machine Reading Comprehension, MRC），又称阅读理解问答，要求机器阅读并理解人类自然语言文本，在此基础上，解答跟文本信息相关的问题。该任务通常被用来衡量机器自然语言理解能力，可以帮助人类从大量文本中快速聚焦相关信息，降低人工信息获取成本，在文本问答、信息抽取、对话系统等领域具有极强的应用价值。近年来，机器阅读理解受到工业界和学术界越来越广泛的关注，是自然语言处理领域的研究热点之一。 

### 1.2 常见任务介绍

 机器阅读理解通过给定上下文，要求机器根据上下文回答问题，来测试机器理解自然语言的程度。常见的任务主要分为完形填空，多项选择，片段抽取和自由回答。而近年来，学界考虑到当前方法的局限性，MRC出现了新的任务，主要有基于知识的机器阅读理解，不可答问题的机器阅读理解，多文本机器阅读理解和对话型问题回答。

#### 1.2.1 完形填空（Cloze Test）

给定上下文![](http://latex.codecogs.com/gif.latex?C)，一个词或实体![](http://latex.codecogs.com/gif.latex?a(a\in{C}))被移除，完形填空任务要求模型使用正确的词或实体进行填空，最大化条件概率![](http://latex.codecogs.com/gif.latex?P(a|C-\{a\}))。

#### 1.2.2 多项选择（Multiple Choice）

给定上下文![](http://latex.codecogs.com/gif.latex?C)，问题![](http://latex.codecogs.com/gif.latex?Q)，候选答案列表![](http://latex.codecogs.com/gif.latex?A=\{a_{1},a_{2},...,a_{n}\})，多项选择任务要求模型从A中选择正确的答案$ a_{i}$，最大化条件概率![](http://latex.codecogs.com/gif.latex?P(a_{i},C,Q,A))。与完形填空任务的区别就是答案不再局限于单词或实体，并且候选答案列表是必须要提供的。

#### 1.2.3 片段抽取（Span Extraction）

给定上下文![](http://latex.codecogs.com/gif.latex?C)和问题![](http://latex.codecogs.com/gif.latex?Q)，其中![](http://latex.codecogs.com/gif.latex?C=\{t_{1},t_{2},...,t_{n}\})，片段抽取任务要求模型从![](http://latex.codecogs.com/gif.latex?C)中抽取连续的子序列![](http://latex.codecogs.com/gif.latex?a=\{t_{i},t_{i+1},...,t_{i+k}\}(1\leq{i}\leq{i+k}\leq{n}))作为正确答案，最大化条件概率![](http://latex.codecogs.com/gif.latex?P(a|C,Q))。

#### 1.2.4 自由回答（ Free Answering）

给定上下文![](http://latex.codecogs.com/gif.latex?C)和问题![](http://latex.codecogs.com/gif.latex?Q)，在自由回答任务中正确答案可能不是C中的一个子序列，![](http://latex.codecogs.com/gif.latex?a\subseteq{C})或![](http://latex.codecogs.com/gif.latex?a\not\subseteq{C})，自由回答任务需要预测正确答案![](http://latex.codecogs.com/gif.latex?a)，并且最大化条件概率![](http://latex.codecogs.com/gif.latex?P(a|C,Q))。

#### 1.2.5 基于知识的机器阅读理解（Knowledge-Based MRC）

 有时候，我们只根据context是无法回答问题的，需要借助外部知识。因此，基于外部知识的MRC应运而生。KBMRC和MRC的不同主要在输入部分，MRC的输入是context和question，而KBMRC的输入是context、question、knowledge。 

<img src="https://pic2.zhimg.com/80/v2-7639ff4b878691078a327a083329f728_720w.jpg" alt="img"/>

相比传统的MRC，KBMRC的挑战在于：

- 相关外部知识检索（Relevant External Knowledge Retrieval）：如何从知识库中找到“用铲子挖洞”这一常识
- 外部知识整合（External Knowledge Integration）： 知识库中结构化的知识如何与非结构化的文本进行融合 

#### 1.2.6 不可答问题的机器阅读理解（MRC with Unanswerable Questions）

 有一个潜在的假设就是MRC任务中正确答案总是存在于给定的上下文中。显然这是不现实的，上下文覆盖的知识是有限的，存在一些问题是无法只根据上下文就可以回答的。因此，MRC系统应该区分这些无法回答的问题。 

<img src="http://5b0988e595225.cdn.sohucs.com/images/20190725/1914f31124a542a682461f1c6cd28b09.jpeg" alt="img" style="zoom: 67%;" />

相比传统的MRC，MRC UQ的挑战在于：

- 不可答问题的判别（Unanswerable Question Detection）： 判断“1937 年条约的名字是什么”这个问题能否根据文章内容进行作答 
- 合理的答案区分（Plausible Answer Discrimination）：避免被 1940 年条约名字这一干扰答案误导 

#### 1.2.7 多文本机器阅读理解（Multi-passage MRC）

 在MRC任务中，相关的段落是预定义好的，这与人类的问答流程矛盾。因为人们通常先提出一个问题，然后再去找所有相关的段落，最后在这些段落中找答案。因此研究学者提出了multi-passage machine reading comprehension，相关数据集有MS MARCO、TriviaQA、SearchQA、Dureader、QUASAR。 

 <img src="https://pic2.zhimg.com/80/v2-440cd6d5068a467954c83340a209880b_720w.jpg" alt="img"/>

相比传统的MRC，MP MRC的挑战在于：

- 海量文件语料的检索（Massive Document Corpus）： 如何从多篇文档中检索到与回答问题相关的文档
- 含噪音的文件检索（Noisy Document Retrieval）：一些文档中可能存在标记答案，但是这些答案与问题可能存在答非所问的情况 
- 无答案（No Answer）
- 多个答案（Multiple Answers）：例如问“美国总统是谁”，特朗普和奥巴马都是可能的答案，但哪一个是正确答案还需要结合语境进行推断 
-  对多条线索进行汇总（Evidence Aggregation）：回答问题的线索可能出现在多篇文档中，需要对其进行总结归纳才能得出正确答案 

#### 1.2.8 对话型问题回答（Conversational Question Answering）

 MRC系统理解了给定段落的语义后回答问题，问题之间是相互独立的。然而，人们获取知识的最自然方式是通过一系列相互关联的问答过程。比如，给定一个问答，A提问题，B回复答案，然后A根据答案继续提问题。这个方式有点类似多轮对话。 

<img src="http://5b0988e595225.cdn.sohucs.com/images/20190725/99fb049c813e4702b540ee1a5edc46d5.jpeg" alt="img" style="zoom:67%;" />

相比传统的MRC，CQA的挑战在于：

- 对谈话历史的利用（Conversational History）：后续的问答过程与之前的问题、答案紧密相关，如何有效利用之前的对话信息 
- 指代消解（Coreference Resolution）： 理解问题 2，必须知道其中的 she 指的是 Jessica 
- 共指解析Coreference Resolution

### 1.3 数据集

 **CNN & Daily Mail**

- Hermann等人于2015年在《Teaching Machines to Read and Comprehend》一文中发布。
- 从CNN和每日邮报上收集了大约一百万条新闻数据作为语料库。
- 通过实体检测等方法将总结和解释性的句子转化为[背景, 问题, 答案]三元组

**CBT**

- 由 Facebook 于 2016 年在《The Goldilocks Principle: Reading Children’s Books with Explicit Memory Representations》一文中发布
- 来自古腾堡项目免费提供的书籍作为语料库
- 由文字段落和相应问题构建

**MCTest**

- 由 Microsoft于 2013年在《MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text》一文中发布
- 开放域机器理解的挑战数据集，有660个阅读理解

**RACE**

- 由Guokun Lai等人在2017年在《RACE: Large-scale ReAding Comprehension Dataset From Examinations》中发布
- 来自中国12-18岁之间的初中和高中英语考试阅读理解，包含28,000个短文、接近100,000个问题。
- 该数据集中的问题中需要推理的比例比其他数据集更高，也就是说，精度更高、难度更大。

**SQuAD**

- 由Pranav Rajpurkar等人在2016年在《SQuAD: 100,000+ Questions for Machine Comprehension of Text》中发布
- 由维基百科的536偏文章上提出的问题组成，包含10万个（问题，原文，答案）三元组，其中每个问题的答案都是一段文本

**NewsQA**

- 由Adam Trischler等人在2016年在《NewsQA: A Machine Comprehension Dataset》中发布
- 由超过12000篇新闻文章和120,000答案组成，每篇文章平均616个单词，每个问题有2～3个答案。

**bAbI**

- 由Jason Weston等人在2016年在《Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks》中发布
- 由若干条文本，1000个训练集问题和1000个测试集问题组成，格式为：
	    ID 文本
	    ID 文本
	    ID 问题 [标签] 答案 [标签] 支持事实的文本ID

**MS MARCO**

- 由Microsoft于 2016年在《 MS MARCO: A Human Generated MAchine Reading COmprehension Dataset》中发布
- 一个大规模英文阅读理解数据集，数据集根据用户在 BING 中输入的真实问题和小娜虚拟助手的真实查询，包含10万个问题和20万篇不重复的文档。

 **Cloze Test** 任务数据集

|      数据集      |                            Paper                             |                             Data                             |
| :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| CNN & Daily Mail | [Teaching Machines to Read and Comprehend](https://arxiv.org/abs/1506.03340) |            [链接](https://cs.nyu.edu/~kcho/DMQA/)            |
|       CBT        | [The Goldilocks Principle: Reading Children’s Books with Explicit Memory Representations](https://arxiv.org/abs/1511.02301) |       [链接](https://research.fb.com/downloads/babi/)        |
|     LAMBADA      |                                                              |                                                              |
|   Who-did-What   |                                                              | [链接](https://tticnlp.github.io/who_did_what/download.html) |
|      CLOTH       |                                                              |                                                              |
|      CliCR       |                                                              |                                                              |

 **Multiple Choice** 任务数据集

| 数据集 |                            Paper                             |                             Data                             |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| MCTest | [MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text](https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/) | [链接](https://github.com/mcobzarenco/mctest/tree/master/data/MCTest) |
|  RACE  | [RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://arxiv.org/abs/1704.04683) |       [链接](https://www.cs.cmu.edu/~glai1/data/race/)       |

 **Span Extraction** 任务数据集

|  数据集  |                            Paper                             |                        Data                         |
| :------: | :----------------------------------------------------------: | :-------------------------------------------------: |
|  SQuAD   | [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) | [链接](https://rajpurkar.github.io/SQuAD-explorer/) |
|  NewsQA  | [NewsQA: A Machine Comprehension Dataset](https://arxiv.org/abs/1611.09830) |      [链接](https://github.com/Maluuba/newsqa)      |
| TriviaQA |                                                              |                                                     |
|  DuoRC   |                                                              |                                                     |

 **Free Answering** 任务数据集

|   数据集    |                            Paper                             |                      Data                       |
| :---------: | :----------------------------------------------------------: | :---------------------------------------------: |
|    bAbI     | [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](https://arxiv.org/abs/1611.09268) | [链接](https://research.fb.com/downloads/babi/) |
|  MS MARCO   | [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268) |  [链接](https://microsoft.github.io/msmarco/)   |
|  SearchQA   |                                                              |                                                 |
| NarrativeQA |                                                              |                                                 |
|  DuReader   |                                                              |                                                 |

### 1.4 SOTA

**SQuAD**

SA-Net on Albert (ensemble)	90.724	EM

SA-Net on Albert (ensemble)	93.011 F1

**官网：**[链接](https://rajpurkar.github.io/SQuAD-explorer/)

**MS MARCO Question Answering**

Multi-doc Enriched BERT Ming Yan of Alibaba Damo NLP   0.540 Rouge-L

Multi-doc Enriched BERT Ming Yan of Alibaba Damo NLP   0.565 Bleu-1

**官网：**[链接](https://microsoft.github.io/msmarco/)

### 1.5 评测标准

- Accuracy（主要用于Cloze Test 和 Multiple Choice）
- Exact Match（主要用于Span Prediction）
- Rouge-L（主要用于Free Answering）
- Bleu（主要用于Free Answering）

## 2 方法总结

### 2.1 基于规则的方法

通过人工制定规则，选取不同的特征 ，基于选取的特征构造并学习一个三元打分函数$f(a, q, d)$，然后选取得分最高的候选语句作为答案。

当前被证明有效的浅层特征主要包括：

- 与答案本身相关，例如：答案在原文中是否出现、出现的频率等信息
- 答案与问题在原文中的关联，例如：答案与问题中的词语在原文中的距离，答案与问题在原文中的窗口序列的N-gram匹配度，答案中的实体与问题中的实体的共现情况等
- 依存语法，一般通过启发式规则将问题与答案组合，然后抽取出依存关系对。同时对原文进行依存句法分析，然后考察问题/答案对的依存句法与原文的依存句法的匹配情况
- 语篇关系，考察与问题相关的多个句子在原文中的语篇关系，例如：一个问题是以Why开头的问句，那么这个问题的多个相关句子在原文之中可能存在因果关系

 除了上述浅层的特征之外，也有一些较为深层次的语义特征被引入到了阅读理解问题当中

-  语义框架匹配，用于考察答案/问题与文章当中的句子的语义框架匹配程度
-  Word Embedding， 例如含有BOW或基于依存树结构的匹配方法

 基于传统特征工程的方法在部分阅读理解任务上能够起到非常好的效果，但是仍然有很多问题不能解决。总的来说，由于大多数传统特征是基于离散的串匹配的，因此在解决表达的多样性问题上显得较为困难。除此之外，由于大多数特征工程的方法都是基于窗口匹配的，因此很难处理多个句子之间的长距离依赖问题。虽然近年来提出了基于多种不同层次窗口的模型可以缓解这一问题，但是由于窗口或者n-gram并不是一个最有效的语义单元，存在语义缺失（缺少部分使语义完整的词）或者噪声（引入与主体语义无关的词）等问题，因此该问题仍然较难解决。 

### 2.2 基于神经网络的方法

  近年来，随着深度学习的兴起，许多基于神经网络的方法被引入到了阅读理解任务中。相比于基于传统特征的方法，在神经网络中，各种语义单元被表示为连续的语义空间上的向量，可以非常有效地解决语义稀疏性以及复述的问题。 当前主流的模型框架主要包括：

- 词向量模块（Embeddings）：传统的词表示（One-hot，Distributed）；预训练上下文表示；添加更多信息（字向量、POS、NER、词频、问题类别）
- 编码模块（Feature Extraction）：RNN（LSTM，GRN）；CNN
- 注意力模块：单路注意力模型、双路注意力模型、自匹配注意力模型
- 答案预测模块（Answer Prediction）：word predictor，option selector，span extractor，answer generator

### 2.3 基于深层语义的图匹配方法

 上述的方法在某些简单的阅读理解任务中能够起到较好的效果。但是对于某些需要引入外部知识进行更深层次推理、几乎不可能仅仅通过相似度匹配得到的结果的阅读理解任务来说，上述方法几乎起不到作用。一个最典型的例子就是Berant等人提出的Biological Processes。该问题需要机器阅读一篇与生化过程有关的文章，并且根据文章回答问题。文中给出的一个例子如图所示。可以看到该问题涉及到大量的知识推理以及理解方面的内容。 

 <img src="http://bbs-10075040.file.myqcloud.com/uploads/images/201610/14/23/YanABAeemB.png" alt="img" style="zoom:80%;" />

 针对上述问题，Berant 等人提出了一种基于图匹配的方法。该方法首先通过类似于语义角色标注的方法，将整篇文章转化成一个图结构。然后将问题与答案组合（称为查询），也转化为一个图结构，最后考虑文章的图结构与查询的图结构之间的匹配度。

### 2.4 小结

本章介绍了在机器阅读理解任务中最常用的三种基本方法：基于特征工程的方法，基于神经网络的方法以及基于深层语义的图匹配方法。三种方法各有侧重，有着不同的应用场景。

基于传统特征的方法在模型结构以及实现上最为简单，在某些特定的数据集上也能起到较好的效果。但是由于特征本身所具有的局限性，该类方法很难处理复述以及远距离依赖问题。

基于深度学习的方法能够很好地处理复述和长距离依赖问题，但是对于某些需要引入外部知识进行更深层次推理、几乎不可能仅仅通过相似度匹配得到结果的任务则无能为力。

基于深层语义的图匹配方法通过在深层次的语义结构中引入人为定义的知识，从而使得模型具有捕捉更深层次语义信息的能力，大大提高了模型的理解以及推理能力。但是由于这类方法对于外部知识的依赖性极强，因此适用范围较窄，可拓展性较弱。

## 3 Paper List

### 3.1 经典论文

| 会议名称  |                           论文名称                           |
| :-------: | :----------------------------------------------------------: |
| NIPS 2015 |           Teaching machines to read and comprehend           |
| ACL 2016  |   Text understanding with the attention sum reader network   |
| ACL 2016  | A Through Examination of the CNN_Daily Mail Reading Comprehension Task |
| ACL 2017  | Attention-over-Attention Neural Networks for Reading Comprehension |
| ICLR 2017 |    Bidirectional Attention Flow for Machine Comprehension    |
| ACL 2017  | Gated Self-Matching Networks for Reading Comprehension and Question Answering |
| ACL 2018  |  Simple and Effective Multi-Paragraph Reading Comprehension  |

### 3.2 近3年顶会论文

| 会议名称  |                           论文名称                           |
| :-------: | :----------------------------------------------------------: |
| AAAI2020  |     SG-Net: Syntax-Guided Machine Reading Comprehension.     |
| AAAI2020  | Generating Well-Formed Answers by Machine Reading with Stochastic Selector Networks. |
| AAAI2020  | A Robust Adversarial Training Approach to Machine Reading Comprehension. |
| AAAI2020  | Multi-Task Learning with Generative Adversarial Training for Multi-Passage Machine Reading Comprehension. |
| AAAI2020  | Distill BERT to Traditional Models in Chinese Machine Reading Comprehension (Student Abstract). |
| AAAI2020  | Assessing the Benchmarking Capacity of Machine Reading Comprehension Datasets. |
| AAAI2020  | A Multi-Task Learning Machine Reading Comprehension Model for Noisy Document (Student Abstract) |
| AAAI2020  | Rception: Wide and Deep Interaction Networks for Machine Reading Comprehension (Student Abstract). |
| AAAI2019  | Read + Verify: Machine Reading Comprehension with Unanswerable Questions. |
| AAAI2019  | Teaching Machines to Extract Main Content for Machine Reading Comprehension. |
| AAAI2018  | Byte-Level Machine Reading Across Morphologically Varied Languages. |
| AAAI2018  | S-Net: From Answer Extraction to Answer Synthesis for Machine Reading Comprehension. |
| EMNLP2019 | Incorporating External Knowledge into Machine Reading for Generative Question Answering. |
| EMNLP2019 |         Cross-Lingual Machine Reading Comprehension.         |
| EMNLP2019 | A Span-Extraction Dataset for Chinese Machine Reading Comprehension. |
| EMNLP2019 | Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning. |
| EMNLP2019 | Towards Machine Reading for Interventions from Humanitarian-Assistance Program Literature. |
| EMNLP2019 | Revealing the Importance of Semantic Retrieval for Machine Reading at Scale. |
| EMNLP2019 | Machine Reading Comprehension Using Structural Knowledge Graph-aware Network. |
| EMNLP2019 | NumNet: Machine Reading Comprehension with Numerical Reasoning. |
| EMNLP2019 | Adversarial Domain Adaptation for Machine Reading Comprehension. |
| ACL 2020  | Explicit Memory Tracker with Coarse-to-Fine Reasoning for Conversational Machine Reading. |
| ACL 2020  | Recurrent Chunking Mechanisms for Long-Text Machine Reading Comprehension. |
| ACL 2020  | A Frame-based Sentence Representation for Machine Reading Comprehension. |
| ACL 2020  |            Machine Reading of Historical Events.             |
| ACL 2020  | A Self-Training Method for Machine Reading Comprehension with Soft Evidence Extraction. |
| ACL 2020  | Enhancing Answer Boundary Detection for Multilingual Machine Reading Comprehension. |
| ACL 2020  | Document Modeling with Graph Attention Networks for Multi-grained Machine Reading Comprehension. |
|  ACL2019  | Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading. |
|  ACL2019  | Explicit Utilization of General Knowledge in Machine Reading Comprehension. |
|  ACL2019  | Enhancing Pre-Trained Language Representations with Rich Knowledge for Machine Reading Comprehension. |
|  ACL2019  | MC\2: Multi-perspective Convolutional Cube for Conversational Machine Reading Comprehension. |
|  ACL2019  | E3: Entailment-driven Extracting and Editing for Conversational Machine Reading. |
|  ACL2019  | Learning to Ask Unanswerable Questions for Machine Reading Comprehension. |
|  ACL2018  | Stochastic Answer Networks for Machine Reading Comprehension. |
|  ACL2018  |        Jack the Reader - A Machine Reading Framework.        |
|  ACL2018  | Multi-Passage Machine Reading Comprehension with Cross-Passage Answer Verification. |
|  ACL2018  | Multi-Relational Question Answering from Narratives: Machine Reading and Reasoning in Simulated Worlds. |
| IJCAI2020 | LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning. |
| IJCAI2020 | [An Iterative Multi-Source Mutual Knowledge Transfer Framework for Machine Reading Comprehension.](https://www.ijcai.org/Proceedings/2020/525) |
| IJCAI2020 | [Asking Effective and Diverse Questions: A Machine Reading Comprehension based Framework for Joint Entity-Relation Extraction.](https://www.ijcai.org/Proceedings/2020/546) |
| IJCAI2018 | [Reinforced Mnemonic Reader for Machine Reading Comprehension.](https://www.ijcai.org/Proceedings/2018/570) |
| CIKM2019  | [Machine Reading Comprehension: Matching and Orders.](https://dl.acm.org/doi/10.1145/3357384.3358139) |
| CIKM2018  | [An Option Gate Module for Sentence Inference on Machine Reading Comprehension.](https://dl.acm.org/doi/10.1145/3269206.3269280) |

## 4 相关链接

## 5 参考资源

[自然语言处理常用数据集](https://zhuanlan.zhihu.com/p/46834868 )

[机器阅读理解综述]( https://zhuanlan.zhihu.com/p/80905984 )

[CIPS青工委学术专栏第6期 | 机器阅读理解任务综述](http://mp.weixin.qq.com/s?__biz=MzIxNzE2MTM4OA==&mid=2665643130&idx=1&sn=5f75f0d4978289caea6c4cb37b0b74c4) 

[神经机器阅读理解最新综述：方法和趋势 ]( https://www.sohu.com/a/329167296_500659 )

