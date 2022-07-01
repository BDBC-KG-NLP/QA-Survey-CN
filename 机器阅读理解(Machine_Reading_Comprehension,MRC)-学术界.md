# MRC调研——学术界

## News！

 - ACL2022、AAAI2022和SIGIR2022 中阅读理解相关工作已经更新。（20220612）
 - 增加Logical Reasoning逻辑推理阅读理解相关数据集和模型介绍（20220608）

## 1.任务

### 1.1 背景

 机器阅读理解（Machine Reading Comprehension, MRC），又称阅读理解问答，要求机器阅读并理解人类自然语言文本，在此基础上，解答跟文本信息相关的问题。该任务通常被用来衡量机器自然语言理解能力，可以帮助人类从大量文本中快速聚焦相关信息，降低人工信息获取成本，在文本问答、信息抽取、对话系统等领域具有极强的应用价值。近年来，机器阅读理解受到工业界和学术界越来越广泛的关注，是自然语言处理领域的研究热点之一。 

### 1.2 常见任务介绍

 机器阅读理解通过给定上下文，要求机器根据上下文回答问题，来测试机器理解自然语言的程度。常见的任务主要分为**完形填空，多项选择，片段抽取和自由回答**。而近年来，学界考虑到当前方法的局限性，MRC出现了新的任务，主要有**基于知识的机器阅读理解，不可答问题的机器阅读理解，多文本机器阅读理解和对话型问题回答**。

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

<div align=center><img src="https://pic2.zhimg.com/80/v2-7639ff4b878691078a327a083329f728_720w.jpg" width=650 alt="img"/></div>
相比传统的MRC，KBMRC的挑战在于：

- **相关外部知识检索**（Relevant External Knowledge Retrieval）：如何从知识库中找到“用铲子挖洞”这一常识
- **外部知识整合**（External Knowledge Integration）： 知识库中结构化的知识如何与非结构化的文本进行融合 

#### 1.2.6 不可答问题的机器阅读理解（MRC with Unanswerable Questions）

 有一个潜在的假设就是MRC任务中正确答案总是存在于给定的上下文中。显然这是不现实的，上下文覆盖的知识是有限的，存在一些问题是无法只根据上下文就可以回答的。因此，MRC系统应该区分这些无法回答的问题。 

<div align=center><img src="http://5b0988e595225.cdn.sohucs.com/images/20190725/1914f31124a542a682461f1c6cd28b09.jpeg" alt="img" width=650" /></div>
相比传统的MRC，MRC UQ的挑战在于：

- **不可答问题的判别**（Unanswerable Question Detection）： 判断“1937 年条约的名字是什么”这个问题能否根据文章内容进行作答 
- **合理的答案区分**（Plausible Answer Discrimination）：避免被 1940 年条约名字这一干扰答案误导 

#### 1.2.7 多文本机器阅读理解（Multi-passage MRC）

 在MRC任务中，相关的段落是预定义好的，这与人类的问答流程矛盾。因为人们通常先提出一个问题，然后再去找所有相关的段落，最后在这些段落中找答案。因此研究学者提出了multi-passage machine reading comprehension。 

 <div align=center><img src="https://pic2.zhimg.com/80/v2-440cd6d5068a467954c83340a209880b_720w.jpg" width=650 alt="img" /></div>
相比传统的MRC，MP MRC的挑战在于：

- **海量文件语料的检索**（Massive Document Corpus）： 如何从多篇文档中检索到与回答问题相关的文档
- **含噪音的文件检索**（Noisy Document Retrieval）：一些文档中可能存在标记答案，但是这些答案与问题可能存在答非所问的情况 
- **无答案**（No Answer）：在文档中没有相应问题的答案
- **多个答案**（Multiple Answers）：例如问“美国总统是谁”，特朗普和奥巴马都是可能的答案，但哪一个是正确答案还需要结合语境进行推断 
-  **对多条线索进行汇总**（Evidence Aggregation）：回答问题的线索可能出现在多篇文档中，需要对其进行总结归纳才能得出正确答案 

#### 1.2.8 对话型问题回答（Conversational Question Answering）

 MRC系统理解了给定段落的语义后回答问题，问题之间是相互独立的。然而，人们获取知识的最自然方式是通过一系列相互关联的问答过程。比如，给定一个问答，A提问题，B回复答案，然后A根据答案继续提问题。这个方式有点类似多轮对话。 

<div align=center><img src="http://5b0988e595225.cdn.sohucs.com/images/20190725/99fb049c813e4702b540ee1a5edc46d5.jpeg" alt="img" width=650 /></div>
相比传统的MRC，CQA的挑战在于：

- **对谈话历史的利用**（Conversational History）：后续的问答过程与之前的问题、答案紧密相关，如何有效利用之前的对话信息 
- **指代消解**（Coreference Resolution）： 理解问题 2，必须知道其中的 she 指的是 Jessica

### 1.3 数据集

####  1.3.1 Cloze Test任务数据集

|      数据集      |                            Paper                             |                             Data                             |
| :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| CNN & Daily Mail | [Teaching Machines to Read and Comprehend](https://arxiv.org/abs/1506.03340) |            [链接](https://cs.nyu.edu/~kcho/DMQA/)            |
|       CBT        | [The Goldilocks Principle: Reading Children’s Books with Explicit Memory Representations](https://arxiv.org/abs/1511.02301) |       [链接](https://research.fb.com/downloads/babi/)        |
|     LAMBADA      | [The LAMBADA dataset: Word prediction requiring a broad discourse context](https://arxiv.org/abs/1606.06031) |          [链接](https://zenodo.org/record/2630551)           |
|   Who-did-What   | [Who did What : A Large-Scale Person-Centered Cloze Dataset](https://www.aclweb.org/anthology/D16-1241/) | [链接](https://tticnlp.github.io/who_did_what/download.html) |
|      CLOTH       | [CLOTH: Large-scale Cloze Test Dataset Designed by Teachers](https://arxiv.org/abs/1711.03225) |      [链接](https://www.cs.cmu.edu/~glai1/data/cloth/)       |
|      CliCR       | [CliCR: A Dataset of Clinical Case Reports for Machine Reading Comprehension](https://arxiv.org/abs/1803.09720) |            [链接](https://github.com/clips/clicr)            |

-   **CNN & Daily Mail**
	- Hermann等人于2015年在《Teaching Machines to Read and Comprehend》一文中发布。
	- 从CNN和每日邮报上收集了大约一百万条新闻数据作为语料库。
	- 通过实体检测等方法将总结和解释性的句子转化为[背景, 问题, 答案]三元组

- **CBT**
	- 由 Facebook 于 2016 年在《The Goldilocks Principle: Reading Children’s Books with Explicit Memory Representations》一文中发布
	- 来自古腾堡项目免费提供的书籍作为语料库
	- 由文字段落和相应问题构建

#### 1.3.2 Multiple Choice任务数据集

| 数据集 |                            Paper                             |                             Data                             |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| MCTest | [MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text](https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/) | [链接](https://github.com/mcobzarenco/mctest/tree/master/data/MCTest) |
|  RACE  | [RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://arxiv.org/abs/1704.04683) |       [链接](https://www.cs.cmu.edu/~glai1/data/race/)       |

-  **MCTest**
	- 由 Microsoft于 2013年在《MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text》一文中发布
	- 开放域机器理解的挑战数据集，有660个阅读理解

- **RACE**
	- 由Guokun Lai等人在2017年在《RACE: Large-scale ReAding Comprehension Dataset From Examinations》中发布
	- 来自中国12-18岁之间的初中和高中英语考试阅读理解，包含28,000个短文、接近100,000个问题
	- 该数据集中的问题中需要推理的比例比其他数据集更高，也就是说，精度更高、难度更大

#### 1.3.3 Span Extraction任务数据集

|  数据集  |                            Paper                             |                        Data                         |
| :------: | :----------------------------------------------------------: | :-------------------------------------------------: |
|  SQuAD   | [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) | [链接](https://rajpurkar.github.io/SQuAD-explorer/) |
|  NewsQA  | [NewsQA: A Machine Comprehension Dataset](https://arxiv.org/abs/1611.09830) |      [链接](https://github.com/Maluuba/newsqa)      |
| TriviaQA | [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://arxiv.org/abs/1705.03551) |   [链接](http://nlp.cs.washington.edu/triviaqa/)    |
|  DuoRC   | [DuoRC: A Large-Scale Dataset for Paraphrased Reading Comprehension](https://arxiv.org/abs/1804.07927) |          [链接](https://duorc.github.io/)           |

-  **SQuAD**
	- 由Pranav Rajpurkar等人在2016年在《SQuAD: 100,000+ Questions for Machine Comprehension of Text》中发布
	- 由维基百科的536偏文章上提出的问题组成，包含10万个（问题，原文，答案）三元组，其中每个问题的答案都是一段文本

- **NewsQA**
	- 由Adam Trischler等人在2016年在《NewsQA: A Machine Comprehension Dataset》中发布
	- 由超过12000篇新闻文章和120,000答案组成，每篇文章平均616个单词，每个问题有2～3个答案

#### 1.3.4 Free Answering任务数据集

|   数据集    |                            Paper                             |                             Data                             |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|    bAbI     | [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks](https://arxiv.org/abs/1611.09268) |       [链接](https://research.fb.com/downloads/babi/)        |
|  MS MARCO   | [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268) |         [链接](https://microsoft.github.io/msmarco/)         |
|  SearchQA   | [SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine](https://arxiv.org/abs/1704.05179) |       [链接](https://github.com/nyu-dl/dl4ir-searchQA)       |
| NarrativeQA | [The NarrativeQA Reading Comprehension Challenge](https://arxiv.org/abs/1712.07040) |        [链接](https://cs.nyu.edu/~kcho/NarrativeQA/)         |
|  DuReader   | [DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications](https://arxiv.org/abs/1711.05073) | [链接](https://ai.baidu.com/broad/subordinate?dataset=dureader) |

- **bAbI**
	- 由Jason Weston等人在2016年在《Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks》中发布
	- 由若干条文本，1000个训练集问题和1000个测试集问题组成，格式为：
		    ID 文本
		    ID 文本
		    ID 问题 [标签] 答案 [标签] 支持事实的文本ID

- **MS MARCO**
	- 由Microsoft于2016年在《 MS MARCO: A Human Generated MAchine Reading COmprehension Dataset》中发布
	- 一个大规模英文阅读理解数据集，数据集根据用户在 BING 中输入的真实问题和小娜虚拟助手的真实查询，包含10万个问题和20万篇不重复的文档

#### 1.3.5  Conversational Question Answering任务数据集

| 数据集 |                            Paper                             |                    Data                     |
| :----: | :----------------------------------------------------------: | :-----------------------------------------: |
|  CoQA  | [CoQA: A Conversational Question Answering Challenge](https://arxiv.org/abs/1808.07042) | [链接](https://stanfordnlp.github.io/coqa/) |
|  QuAC  | [QuAC : Question Answering in Context](https://arxiv.org/abs/1808.07036) |                  [链接]()                   |

- **CoQA**
	- 由Siva Reddy等人于2018年在《CoQA: A Conversational Question Answering Challenge》中国发布
	- 由关于不同领域文章的一组组对话式问答构成的大型数据集，问题非常简短。在对话式问答中，首个问题后的每个问题都是基于前序对话展开的。 
	- 与SQuAD相比， CoQA具有多轮问答的“对话”属性，而且机器的回答形式也更加自由，以确保对话的自然流畅。 
- **QuAC**
	- 由Eunsol Choi等人于2018年在《QuAC : Question Answering in Context》中发布
	- 其中含有大量的How还有what was等主导的问题，所以数据集的预测难度会比以往的数据集要高
	- 问题开放性比较强，答案的长度也长于SQuAD。SQuAD回答的平均长度为3个词，DuAC的是15个词。

#### 1.3.6 **Unanswerable Questions** 任务数据集

|  数据集   |                            Paper                             |                             Data                             |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| SQuAD 2.0 | [Know What You Don’t Know: Unanswerable Questions for SQuAD](https://arxiv.org/pdf/1806.03822.pdf) | [链接](https://worksheets.codalab.org/worksheets/0x9a15a170809f4e2cb7940e1f256dee55/) |

- 由Pranav Rajpurkar等人在2018年提出，并获得 ACL 2018 最佳论文奖的论文

- 新增了超过五万个新增的、由人类众包者对抗性地设计的无法回答的问题

- 要求模型不仅能够给出可回答问题的答案，还要识别出文本中没有知识支持的问题，并拒绝回答这些问题

  

#### 1.3.7 **Multi-Passage Machine Reading Comprehension**任务数据集

|  数据集  |                            Paper                             |                          Data                          |
| :------: | :----------------------------------------------------------: | :----------------------------------------------------: |
| QAngaroo | [Constructing Datasets for Multi-hop Reading Comprehension Across Documents](https://arxiv.org/pdf/1710.06481.pdf) | [链接](https://qangaroo.cs.ucl.ac.uk/leaderboard.html) |
| HotpotQA | [HOTPOTQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://aclanthology.org/D18-1259/) |          [链接](https://hotpotqa.github.io/)           |
| TriviaQA | [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](http://nlp.cs.washington.edu/triviaqa/docs/triviaQA.pdf?continueFlag=1de3221ee66021ccb578e73c8424f4df) |     [链接](http://nlp.cs.washington.edu/triviaqa/)     |

- QAngaroo是由伦敦大学学院在2017年发布的多文档阅读理解数据集。最大的特点是问题的答案并不能从一个段落中单独得出，其线索分散在多个段落，所以答案需要通过推理得出。因此，需要模型利用多跳推理（multi-hop reasoning）获得答案。
- HotpotQA是2018年由多家高校与Google公司等研究机构共同推出的多文档阅读理解数据集。共包含11万个问题和相关的维基百科段落，采用区间式答案。通过一些用户交互设计保证他们可以提问出基于两个选段进行多步推理才能得到答案的问题。
- TriviaQA是2017年ACL 中提出的多轮推理的阅读理解数据集。其特点有：（1）数据集中的文本的句法比较复杂，需要很多复合信息；（2）数据集中的文本的语法和用词也比较复杂，简单的文本匹配方法可能会失效；（3）数据集中的信息常常跨过多句，为了得到答案需要多级推理（cross sentences）

#### 1.3.8 Logical Reasoning 任务数据集

| 数据集 |                            Paper                             |                       Data                       |
| :----: | :----------------------------------------------------------: | :----------------------------------------------: |
| ReClor | [ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning](https://arxiv.org/abs/2002.04326) |         [链接](https://whyu.me/reclor/)          |
| LogiQA | [LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning](https://aclanthology.org/D18-1259/) | [链接](https://github.com/lgw863/LogiQA-dataset) |

- ReClor是一个四选一的单项选择问答数据集，其来自于美国研究生管理科入学考试（GMAT）和法学院入学考试（LSAT）经过筛选、过滤得到6138条考察逻辑推理能力的数据。ReClor来自侧重考察逻辑推理的考试，只有基于篇章、问题和选项进行逻辑推理和分析才能得到正确的答案。
- LogiQA是一个四选一的单项选择问答数据集，数据来自于中国的国家公务员考试题目，其旨在考察公务员候选人的批判性思维和解决问题的能力。经过专业的英文使用者由中文翻译到英文，经过筛选、过滤后得到8678条数据。针对输入的问题、篇章和四个选项，模型需要根据问题和篇章找出唯一正确的选项作为答案。

### 1.4 SOTA

- **SQuAD**
  - SA-Net on Albert (ensemble)	**90.724	EM**
  - SA-Net on Albert (ensemble)	**93.011     F1**
  - **排行榜：**[链接](https://rajpurkar.github.io/SQuAD-explorer/)
- **MS MARCO Question Answering**
  - Multi-doc Enriched BERT Ming Yan of Alibaba Damo NLP   **0.540     Rouge-L**
  - Multi-doc Enriched BERT Ming Yan of Alibaba Damo NLP   **0.565     Bleu-1**
  - **排行榜：**[链接](https://microsoft.github.io/msmarco/)
- **RACE Reading Comprehension Dataset**
  - **RACE数据集**	 Megatron-BERT (ensemble)	 **90.9	Accuracy**
  - **RACE-M数据集**	 Megatron-BERT (ensemble)	 **93.1	Accuracy**
  - **RACE-H数据集**	  Megatron-BERT (ensemble)	 **90.0	Accuracy**
  - **排行榜：**[链接](http://www.qizhexie.com//data/RACE_leaderboard )
- **CoQA**
  - **In-domain**   TR-MT (ensemble) 	 **91.5	F1**
  - **Out-of-domain**   RoBERTa + AT + KD (ensemble)	  **89.2	F1**
  - **Overal**l   RoBERTa + AT + KD (ensemble)	  **90.7	F1**
  - **排行榜：**[链接]( https://stanfordnlp.github.io/coqa/ )
- **QuAC**
  -  TR-MT (ensemble)    **74.4     F1**
  -  TR-MT (ensemble)    **71.3     HEQQ**
  -  TR-MT (ensemble)    **13.6     HEQD**
  - **排行榜：**[链接](https://quac.ai/?qqdrsign=03085 )
- **Dureader**
  - Cross-Reader	**64.9	ROUGE-L**
  - Cross-Reader  **61.77	BLEU-4**
  - **排行榜：**[链接](https://ai.baidu.com/broad/subordinate?dataset=dureader)


- **HotpotQA**
  - Ans	FE2H on ALBERT	**71.89**	**EM**
  - Sup     FE2H on ALBERT    **64.98**    **EM**
  - Joint    FE2H on ALBERT    **50.04**   **EM**
  - **排行榜：**[链接](https://hotpotqa.github.io/)
- **ReClor**

  -  easy	XLNet-large	 **75.7	Accuracy**
  - standard 	XLNet-large	**56.0	Accuracy**
  - hard	XLNet-large	 **40.5	Accuracy**
  - 排行版：[链接](https://paperswithcode.com/paper/reclor-a-reading-comprehension-dataset-1)


### 1.5 评测标准

- **Accuracy**（主要用于Cloze Test 和 Multiple Choice）
- **Exact Match**（主要用于Span Prediction）
- **Rouge-L**（主要用于Free Answering）
- **Bleu**（主要用于Free Answering）

## 2 方法总结

### 2.1 基于规则的方法

通过人工制定规则，选取不同的特征 ，基于选取的特征构造并学习一个三元打分函数$f(a, q, d)$，然后选取得分最高的候选语句作为答案。

当前被证明有效的浅层特征主要包括：

- **与答案本身相关**，例如：答案在原文中是否出现、出现的频率等信息
- **答案与问题在原文中的关联**，例如：答案与问题中的词语在原文中的距离，答案与问题在原文中的窗口序列的N-gram匹配度，答案中的实体与问题中的实体的共现情况等
- **依存语法**，一般通过启发式规则将问题与答案组合，然后抽取出依存关系对。同时对原文进行依存句法分析，然后考察问题/答案对的依存句法与原文的依存句法的匹配情况
- **语篇关系**，考察与问题相关的多个句子在原文中的语篇关系，例如：一个问题是以Why开头的问句，那么这个问题的多个相关句子在原文之中可能存在因果关系

 除了上述浅层的特征之外，也有一些较为深层次的语义特征被引入到了阅读理解问题当中

-  **语义框架匹配**，用于考察答案/问题与文章当中的句子的语义框架匹配程度
-  **Word Embedding**， 例如含有BOW或基于依存树结构的匹配方法

 基于传统特征工程的方法在部分阅读理解任务上能够起到非常好的效果，但是仍然有很多问题不能解决。

- 大多数传统特征是基于离散的串匹配的，在解决**表达的多样性**问题上显得较为困难。
- 大多数特征工程的方法都是基于窗口匹配的，因此很难处理**多个句子之间的长距离依赖**问题。虽然近年来提出了基于多种不同层次窗口的模型可以缓解这一问题，但是由于窗口或者n-gram并不是一个最有效的语义单元，存在语义缺失（缺少部分使语义完整的词）或者噪声（引入与主体语义无关的词）等问题，因此该问题仍然较难解决。 

### 2.2 基于神经网络的方法

  近年来，随着深度学习的兴起，许多基于神经网络的方法被引入到了阅读理解任务中。相比于基于传统特征的方法，在神经网络中，各种语义单元被表示为连续的语义空间上的向量，可以非常有效地解决语义稀疏性以及复述的问题。 当前主流的模型框架主要包括：

- **词向量模块**（Embeddings）：传统的词表示（One-hot，Distributed）；预训练上下文表示；添加更多信息（字向量、POS、NER、词频、问题类别）
- **编码模块**（Feature Extraction）：RNN（LSTM，GRN）；CNN
- **注意力模块**：单路注意力模型、双路注意力模型、自匹配注意力模型
- **答案预测模块**（Answer Prediction）：word predictor，option selector，span extractor，answer generator

### 2.3 基于深层语义的图匹配方法

 上述的方法在某些简单的阅读理解任务中能够起到较好的效果。但是对于某些需要引入外部知识进行更深层次推理、几乎不可能仅仅通过相似度匹配得到的结果的阅读理解任务来说，上述方法几乎起不到作用。一个最典型的例子就是Berant等人提出的Biological Processes。该问题需要机器阅读一篇与生化过程有关的文章，并且根据文章回答问题。文中给出的一个例子如图所示。可以看到该问题涉及到大量的知识推理以及理解方面的内容。 

 <div align=center><img src="http://bbs-10075040.file.myqcloud.com/uploads/images/201610/14/23/YanABAeemB.png" alt="img" width=400  /></div>
 针对上述问题，Berant 等人提出了一种基于图匹配的方法。该方法首先通过类似于语义角色标注的方法，将整篇文章转化成一个图结构。然后将问题与答案组合（称为查询），也转化为一个图结构，最后考虑文章的图结构与查询的图结构之间的匹配度。

​	但是语篇关系能否充分表示基于逻辑关系的符号推理有待商榷，且图结构稀疏，长路径较多限制了GNN模型中节点与节点间的消息传递。为解决上述问题，**AdaLoGN**采用有向的文本逻辑图(Text Logic Graph，TLG)，使用符号化的推理规则对原始语篇图进行扩展，基于已知推理得到未知的逻辑关系。**Logiformer**则提出基于篇章分别构建逻辑图和句法图，使用一个两支的graph transformer网络来从两个角度建模长距离依赖。

### 2.4 基于预训练模型的方法

​	近年来NLP领域通过大量通用领域数据进行训练，诞生了一批如ELMO、GPT、BERT、ENRIE等优秀的预训练语言模型。在具体的阅读理解任务时，可以通过进行领域微调、数据微调、任务微调，来把学到的句子特征信息应用到具体的任务。

 <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey-CN/blob/master/image/MRC/基于预训练的MRC微调结构图.png" alt="img" width=400  /></div>

​	如图为基于BERT模型的预训练任务和机器阅读理解的微调结构，左侧是预训练过程，经过预训练得到较好的模型参数。右侧是用于机器阅读理解的微调示意图，该神经网络的参数由预训练得到的模型进行初始化，将问题与包含答案的上下文经过编码输入神经网络，获得答案的区间预测。

​	基于一些启发式规则捕捉大规模文本语料中存在的逻辑关系，针对其设计相应的预训练任务，对已有的预训练语言模型进行二次预训练，从而增强预训练语言模型的逻辑推理能力，已成为模型训练的一重要途径。LogiGAN将视线转向了生成式模型（T5）以及包括逻辑推理MRC在内的多种需要推理能力任务上，通过对MLM预训练任务进行改进来强化模型的逻辑推理能力，并引入验证器（verifier）来为生成式模型提供额外反馈。

本章介绍了在机器阅读理解任务中最常用的三种基本方法：基于特征工程的方法，基于神经网络的方法以及基于深层语义的图匹配方法。三种方法各有侧重，有着不同的应用场景。

- **基于传统特征的方法**在模型结构以及实现上最为简单，在某些特定的数据集上也能起到较好的效果。但是由于特征本身所具有的局限性，该类方法很难处理复述以及远距离依赖问题。
- **基于深度学习的方法**能够很好地处理复述和长距离依赖问题，但是对于某些需要引入外部知识进行更深层次推理、几乎不可能仅仅通过相似度匹配得到结果的任务则无能为力。
- **基于深层语义的图匹配方法**通过在深层次的语义结构中引入人为定义的知识，从而使得模型具有捕捉更深层次语义信息的能力，大大提高了模型的理解以及推理能力。但是由于这类方法对于外部知识的依赖性极强，因此适用范围较窄，可拓展性较弱。
- **基于预训练模型的方法**通过在大量文本上进行预训练，得到强健的预训练语言模型，能够捕获句子间更深层次的关联关系。在做机器阅读理解任务时，只需要设计适合具体任务的网络拼接到预训练模型网络上进行微调。

## 3 Paper List

### 3.1 经典论文

| 会议名称  |                           论文名称                           |
| :-------: | :----------------------------------------------------------: |
| NIPS 2015 | [Teaching machines to read and comprehend](https://arxiv.org/abs/1506.03340) |
| ACL 2016  | [Text understanding with the attention sum reader network](https://arxiv.org/abs/1603.01547) |
| ACL 2016  | [A Through Examination of the CNN_Daily Mail Reading Comprehension Task](https://arxiv.org/abs/1606.02858) |
| ACL 2017  | [Attention-over-Attention Neural Networks for Reading Comprehension](https://arxiv.org/abs/1607.04423) |
| ICLR 2017 | [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603) |
| ACL 2017  | [Gated Self-Matching Networks for Reading Comprehension and Question Answering](https://www.aclweb.org/anthology/P17-1018) |
| ACL 2018  | [Simple and Effective Multi-Paragraph Reading Comprehension](https://arxiv.org/abs/1710.10723) |

### 3.2 近3年技术类论文

| 会议名称  |                           论文名称                           |
| :-------: | :----------------------------------------------------------: |
|  ACL2022  | [Modeling Multi-hop Question Answering as Single Sequence Prediction](https://arxiv.org/pdf/2205.09226.pdf) |
|  ACL2022  | [AdaLoGN: Adaptive Logic Graph Network for Reasoning-Based Machine Reading Comprehension ](https://arxiv.org/abs/2203.08992) |
|  ACL2022  | [Deep Inductive Logic Reasoning for Multi-Hop Reading Comprehension](https://aclanthology.org/2022.acl-long.343/) |
|  ACL2022  | [MultiHiertt: Numerical Reasoning over Multi Hierarchical Tabular and Textual Data](https://arxiv.org/abs/2206.01347) |
|  ACL2022  | [Learning Disentangled Semantic Representations for Zero-Shot CrossLingual Transfer in Multilingual Machine Reading Comprehension](https://arxiv.org/pdf/2204.00996.pdf) |
|  ACL2022  | [Lite Unified Modeling for Discriminative Reading Comprehension](https://arxiv.org/pdf/2203.14103.pdf) |
|  ACL2022  | [Modeling Temporal-Modal Entity Graph for Procedural Multimodal Machine Comprehension](https://arxiv.org/pdf/2204.02566.pdf) |
|  ACL2022  | [Improving Machine Reading Comprehension with Contextualized Commonsense Knowledge ](https://arxiv.org/abs/2009.05831) |
|  ACL2021  | [REPT: Bridging Language Models and Machine Reading Comprehension via Retrieval-Based Pre-training](https://arxiv.org/pdf/2105.04201.pdf) |
|  ACL2021  | [Knowledge-Empowered Representation Learning for Chinese Medical Reading Comprehension: Task, Model and Resources](https://arxiv.org/pdf/2008.10327.pdf) |
|  ACL2021  | [Benchmarking Robustness of Machine Reading Comprehension Models](https://arxiv.org/abs/2004.14004) |
| ACL 2020  | [Explicit Memory Tracker with Coarse-to-Fine Reasoning for Conversational Machine Reading](https://arxiv.org/abs/2005.12484) |
| ACL 2020  | [Recurrent Chunking Mechanisms for Long-Text Machine Reading Comprehension.](https://arxiv.org/abs/2005.08056) |
| ACL 2020  | [A Frame-based Sentence Representation for Machine Reading Comprehension.](https://www.aclweb.org/anthology/2020.acl-main.83) |
| ACL 2020  | [Machine Reading of Historical Events.](https://www.aclweb.org/anthology/2020.acl-main.668) |
| ACL 2020  | [A Self-Training Method for Machine Reading Comprehension with Soft Evidence Extraction.](https://arxiv.org/abs/2005.05189) |
| ACL 2020  | [Enhancing Answer Boundary Detection for Multilingual Machine Reading Comprehension.](https://arxiv.org/abs/2004.14069) |
| ACL 2020  | [Document Modeling with Graph Attention Networks for Multi-grained Machine Reading Comprehension.](https://arxiv.org/abs/2005.05806) |
| NAACL2022 | [Topological Information Enhanced Structural Reading Comprehension on Web Pages](https://arxiv.org/pdf/2205.06435.pdf) |
| NAACL2022 | [Understand before Answer: Improve Temporal Reading Comprehension via Precise Question Understanding](https://openreview.net/pdf?id=BOWeq59WHZc) |
| NAACL2022 | [Cooperative Self-training of Machine Reading Comprehension](https://openreview.net/pdf?id=rExe7FqZBWc) |
| NAACL2022 | Continual Machine Reading Comprehension via Uncertainty-aware Fixed Memory and Adversarial Domain Adaptation |
| NAACL2022 | To Answer or Not To Answer? Improving Machine Reading Comprehension Model with Span-based Contrastive Learning |
| NAACL2022 | [Compositional Task-Oriented Parsing as Abstractive Question Answering](https://arxiv.org/pdf/2205.02068.pdf) |
| SIGIR2022 | [Logiformer: A Two-Branch Graph Transformer Network for Interpretable Logical Reasoning](https://arxiv.org/abs/2205.00731) |
| SIGIR2022 | [QUASER: Question Answering with Scalable Extractive Rationalization](https://openreview.net/pdf?id=BhMiwEP5ZM7) |
| SIGIR2022 |  Detecting Frozen Phrases in Open-Domain Question Answering  |
| SIGIR2022 |  PTAU: Prompt Tuning for Attributing Unanswerable Questions  |
| SIGIR2022 | [Answering Count Query with Explanatory Evidence](https://arxiv.org/pdf/2204.05039.pdf) |
| AAAI2022  | [Zero-Shot Cross-Lingual Machine Reading Comprehension via Inter-Sentence Dependency Graph](https://arxiv.org/abs/2112.00503) |
| AAAI2022  | [From Good to Best: Two-Stage Training for Cross-lingual Machine Reading Comprehension ](https://arxiv.org/abs/2112.04735) |
| AAAI2022  | [Block-Skim: Efficient Question Answering for Transformer](https://arxiv.org/abs/2112.08560) |
| AAAI2021  | [Semantics Altering Modifications for Evaluating Comprehension in Machine Reading](https://www.aaai.org/AAAI21Papers/AAAI-704.SchlegelV.pdf) |
| AAAI2021  | [VisualMRC: Machine Reading Comprehension on Document Images](https://arxiv.org/pdf/2101.11272) |
| AAAI2021  | [A Bidirectional Multi-Paragraph Reading Model for Zero-Shot Entity Linking](https://www.aaai.org/AAAI21Papers/AAAI-6269.TangH.pdf) |
| AAAI2021  | [Bidirectional Machine Reading Comprehension for Aspect Sentiment Triplet Extraction](https://arxiv.org/pdf/2103.07665.pdf) |
| AAAI2020  | [SG-Net: Syntax-Guided Machine Reading Comprehension.](https://arxiv.org/abs/1908.05147) |
| AAAI2020  | [Generating Well-Formed Answers by Machine Reading with Stochastic Selector Networks.](https://aaai.org/ojs/index.php/AAAI/article/view/6238) |
| AAAI2020  | [A Robust Adversarial Training Approach to Machine Reading Comprehension.](https://aaai.org/ojs/index.php/AAAI/article/view/6357/6213) |
| AAAI2020  | [Multi-Task Learning with Generative Adversarial Training for Multi-Passage Machine Reading Comprehension.](https://aaai.org/ojs/index.php/AAAI/article/view/6396) |
| AAAI2020  | [Distill BERT to Traditional Models in Chinese Machine Reading Comprehension (Student Abstract).](https://aaai.org/ojs/index.php/AAAI/article/view/7223/7077) |
| AAAI2020  | [Assessing the Benchmarking Capacity of Machine Reading Comprehension Datasets.](https://arxiv.org/abs/1911.09241) |
| AAAI2020  | [A Multi-Task Learning Machine Reading Comprehension Model for Noisy Document (Student Abstract)](https://www.aaai.org/ojs/index.php/AAAI/article/view/7254) |
| AAAI2020  | [Rception: Wide and Deep Interaction Networks for Machine Reading Comprehension (Student Abstract).](https://www.aaai.org/ojs/index.php/AAAI/article/view/7266) |
| AAAI2019  | [Read + Verify: Machine Reading Comprehension with Unanswerable Questions.](https://arxiv.org/abs/1808.05759) |
| AAAI2019  | [Teaching Machines to Extract Main Content for Machine Reading Comprehension.](https://www.aaai.org/ojs/index.php/AAAI/article/view/5123) |
| AAAI2018  | [Byte-Level Machine Reading Across Morphologically Varied Languages.](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16605) |
| AAAI2018  | [S-Net: From Answer Extraction to Answer Synthesis for Machine Reading Comprehension.](https://arxiv.org/abs/1706.04815) |
| EMNLP2021 | [Interactive Machine Comprehension with Dynamic Knowledge Graphs](https://aclanthology.org/2021.emnlp-main.540.pdf) |
| EMNLP2021 | [Summarize-then-Answer: Generating Concise Explanations for Multi-hop Reading Comprehension](https://aclanthology.org/2021.emnlp-main.490.pdf) |
| EMNLP2021 | [Learning with Instance Bundles for Reading Comprehension](https://aclanthology.org/2021.emnlp-main.584.pdf) |
| EMNLP2021 | [Enhancing Multiple-choice Machine Reading Comprehension by Punishing Illogical Interpretations](https://aclanthology.org/2021.emnlp-main.295.pdf) |
| EMNLP2021 | [Smoothing Dialogue States for Open Conversational Machine Reading](https://aclanthology.org/2021.emnlp-main.299.pdf) |
| EMNLP2021 | [Mitigating False-Negative Contexts in Multi-Document Question Answering with Retrieval Marginalization](https://aclanthology.org/2021.emnlp-main.497.pdf) |
| EMNLP2019 | [Incorporating External Knowledge into Machine Reading for Generative Question Answering.](https://arxiv.org/abs/1909.02745) |
| EMNLP2019 | [Cross-Lingual Machine Reading Comprehension.](https://arxiv.org/abs/1909.00361) |
| EMNLP2019 | [A Span-Extraction Dataset for Chinese Machine Reading Comprehension.](https://arxiv.org/abs/1810.07366) |
| EMNLP2019 | [Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning.](https://arxiv.org/abs/1909.00277) |
| EMNLP2019 | [Towards Machine Reading for Interventions from Humanitarian-Assistance Program Literature.](https://www.aclweb.org/anthology/D19-1680) |
| EMNLP2019 | [Revealing the Importance of Semantic Retrieval for Machine Reading at Scale.](https://arxiv.org/abs/1909.08041) |
| EMNLP2019 | [Machine Reading Comprehension Using Structural Knowledge Graph-aware Network.](https://www.aclweb.org/anthology/D19-1602) |
| EMNLP2019 | [NumNet: Machine Reading Comprehension with Numerical Reasoning.](https://arxiv.org/abs/1910.06701) |
| EMNLP2019 | [Adversarial Domain Adaptation for Machine Reading Comprehension.](https://arxiv.org/abs/1908.09209) |
| IJCAI2020 | [LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning.](https://arxiv.org/abs/2007.08124) |
| IJCAI2020 | [An Iterative Multi-Source Mutual Knowledge Transfer Framework for Machine Reading Comprehension.](https://www.ijcai.org/Proceedings/2020/525) |
| IJCAI2020 | [Asking Effective and Diverse Questions: A Machine Reading Comprehension based Framework for Joint Entity-Relation Extraction.](https://www.ijcai.org/Proceedings/2020/546) |

### 3.3 近三年分析类论文

| 会议名称  | 论文名称                                                     |
| --------- | ------------------------------------------------------------ |
| ACL2022   | [What Makes Reading Comprehension Questions Difficult?](https://arxiv.org/pdf/2203.06342.pdf) |
| NAACL2022 | [On the Robustness of Reading Comprehension Models to Entity Renaming](https://arxiv.org/pdf/2110.08555.pdf) |
| EMNLP2021 | [Connecting Attributions and QA Model Behavior on Realistic Counterfactuals](https://aclanthology.org/2021.emnlp-main.447.pdf) |
| ACL2021   | [Why Machine Reading Comprehension Models Learn Shortcuts?](https://arxiv.org/pdf/2106.01024.pdf) |



## 4 相关链接

[RACE数据集上各个模型文章的笔记]( https://zhuanlan.zhihu.com/p/62898980 )

[自然语言处理常见数据集、论文最全整理分享]( https://zhuanlan.zhihu.com/p/56144877 )

[资源：10份机器阅读理解数据集]( https://www.jiqizhixin.com/articles/2017-09-21-7 )

## 5 参考资源

[自然语言处理常用数据集](https://zhuanlan.zhihu.com/p/46834868 )

[机器阅读理解综述]( https://zhuanlan.zhihu.com/p/80905984 )

[CIPS青工委学术专栏第6期 | 机器阅读理解任务综述](http://mp.weixin.qq.com/s?__biz=MzIxNzE2MTM4OA==&mid=2665643130&idx=1&sn=5f75f0d4978289caea6c4cb37b0b74c4) 

[神经机器阅读理解最新综述：方法和趋势 ]( https://www.sohu.com/a/329167296_500659 )

