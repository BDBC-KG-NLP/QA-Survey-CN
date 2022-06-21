# TQA调研——学术界

## 目录

## 1. 任务

### 1.1. 背景

 世界上许多信息都是以表格形式存储的，这些表格见诸于网络、数据库或文件中。它们包括消费产品的技术规格、金融和国家发展统计数据、体育赛事结果等等。目前，要想找到问题的答案，人们仍需以人工方式查找这些表格，或使用能提供特定问题（比如关于体育赛事结果的问题）的答案的服务。如果可通过自然语言来查询这些信息，那么取用这些信息会容易很多。 

### 1.2. 任务定义

#### 1.2.1 概述

 目前，与表格相关的自然语言处理研究刚刚起步，方法尚未成熟，对应的标注数据集也相对有限。我们按照MSRA的相关工作，将任务定义为以下五类。

<div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey-CNblob/master/image/TQA_X_1.png" alt="img" width=650" /></div>

- 表格检索：从表格集合中找到与输入问题最相关的表格
- 语义解析：将自然语言问题转换成可被机器理解的语义表示（meaning representation，一般指SQL语句），在表格中执行该表示即可获得答案
- 问题生成：可看作语义解析的逆过程，能减轻语义解析器对大量标注训练数据的依赖
- 对话：主要用于多轮对话场景的语义解析任务，需有效解决上下文中的省略和指代现象
- 文本生成：使用自然语言描述表格中（如给定的一行）的内容

 **表格问答 **(table-based QA) 就是针对一个**自然语言问题**，根据表格内容给出答案。表格由M行N列数据组成。每一行由N个表格单元构成,表示一条信息记录。每一列由M个表格单元构成,同一列中的所有表格单元具有相同的类型。 针对上面五项任务，表格问答主要涉及的是**表格检索**和**语义解析**两个模块。检索模块在文档问答里也有，文档问答的检索模块是为了找到和问题相关的文档，而表格问答里检索模块的目标则是找到和问题相关的表格。这个模块只有在涉及大量表格（例如搜索引擎）的时候才会用到。

#### 1.2.2 表格检索 (Retrieval)

对于给定的自然语言q和给定的表格全集T={T1, T2, .., Tn}，表格检索任务的目的是从T中找到与q内容最相关的表格，如下图所示。每个表格通常由三部分构成：表头/列名（table header）、表格单元（table cell）和表格标题（table caption）。

<div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey-CN/blob/master/image/TQA_X_2.png" alt="img" width=650" /></div>

#### 1.2.3 答案生成 (Answer Generation)

给定输入问题Q和表格T，答案生成负责基于T生成Q所对应的答案。在该任务中，Q的答案既可以是T中包含的一个或多个表格单元，也可以是T中推理出来的、并未出现的值。如果将表格看作一个微型知识图谱，答案生成可以看作是一种特殊的知识图谱问答任务。基于表格的答案生成方法大致可以分为三类：基于答案排序的方法，基于语义分析的方法，基于神经网络的方法。以下所示的是基于语义分析的方法的一个示例。

<div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey-CN/blob/master/image/TQA_X_3.png" alt="img" width=650" /></div>

### 1.3. 评测标准

- ![](http://latex.codecogs.com/gif.latex?Accuracy@1/Precision@1)（主要用于评估查询语句执行后答案的质量）
- ![](http://latex.codecogs.com/gif.latex?Accuracy_{lf})   (测量与正确SQL查询具有完全匹配的字符串的生成SQL查询的百分比)
- <img src="http://latex.codecogs.com/gif.latex?F1"  />（主要用于评估SQL关键词匹配的效果）
- ![](http://latex.codecogs.com/gif.latex?Recall)（主要用于多答案问题的答案查询效果）

### 1.4. 数据集

#### 1.4.1 数据集总览

|       数据集       | 在统计论文中出现次数 |
| :----------------: | :------------------: |
| WikiTableQuestions |          8           |
|      WikiSQL       |          4           |
|       Spider       |          3           |
|       TabMCQ       |          3           |
|   Sequential QA    |          1           |

#### 1.4.2 数据集介绍

####  WikiTableQuestions

- **相关论文**：Compositional semantic parsing on semi-structured tables
- **下载链接**：[链接](https://ppasupat.github.io/WikiTableQuestions/)
- **说明**：该数据集由斯坦福在2015年发布,主要针对表格问答任务。该数据集总共包含22033条人工标注的<问题,表格,答案>三元组，任务的目标是基于给定表格和问题,生成问题对应的答案。 该数据集中的表格来自英文维基百科，共2108个。Amazon Mechanical Turk上的标注人员针对每个表格编写问题,并基于该表格标注问题对应的答案。

| Method                                  | ACC(DEV) | ACC(TEST) | P@1(TEST) | 论文                                                         | 年份 | Code                                                         |
| --------------------------------------- | -------- | --------- | --------- | ------------------------------------------------------------ | ---- | ------------------------------------------------------------ |
| Iterative Search                        | 85.4     | 82.4      |           | [Iterative Search for Weakly Supervised Semantic Parsing](https://www.aclweb.org/anthology/N19-1273.pdf) | 2019 | https://github.com/allenai/iterative-search-semparse         |
| MeRL                                    |          | 46.9      |           | [Learning to Generalize from Sparse and Underspecified Rewards](https://arxiv.org/pdf/1902.07198.pdf) | 2019 | https://github.com/google-research/google-research/tree/master/meta_reward_learning |
| CNN-FC-BILIN (15 ensemble models）      |          |           | 38.7      | [Neural Multi-step Reasoning for Question Answering on Semi-structured Tables](https://link.springer.com/chapter/10.1007%2F978-3-319-76941-7_52) | 2018 | https://github.com/dalab/neural_qa                           |
| Ensemble of 15 Neural Programmer models | 37.5     | 37.7      |           | [Learning a natural language interface with neural programmer](https://openreview.net/pdf?id=ry2YOrcge) | 2017 | https://github.com/saraswat/NeuralProgrammerAsProbProg       |

####  WikiSQL

- **相关论文**：Seq2sql: Generating structured queries from natural language using reinforcement learning
- **下载链接**：[链接](https://github.com/salesforce/WikiSQL)
- **说明**：该数据集使用从Wikipedia中提取的表，并带有程序（SQL）注释。数据集中共有24,241个表和80,654个问题程序对，分为训练/开发/测试集。与WIKITABLEQUESTIONS相比，语义更简单，因为其中的SQL使用较少的运算符（列选择，聚合和条件）。

| Method                   | ACC  | 论文                                                         | 年份 | Code                                       |
| ------------------------ | ---- | ------------------------------------------------------------ | ---- | ------------------------------------------ |
| NL2SQL-BERT              | 97.0 | [ Content Enhanced BERT-based Text-to-SQL Generation](https://arxiv.org/pdf/1910.07179v5.pdf) | 2019 | https://github.com/guotong1988/NL2SQL-RULE |
| SQLNet                   | 90.3 | [ SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning](https://arxiv.org/pdf/1711.04436v1.pdf) | 2017 | https://github.com/xiaojunxu/SQLNet        |
| TAPAS (fully-supervised) | 86.4 | [TaPas: Weakly Supervised Table Parsing via Pre-training](https://arxiv.org/pdf/2004.02349.pdf) | 2020 | https://github.com/google-research/tapas   |

#### Spider

- **相关论文**：Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task

- **下载链接**：[链接](https://yale-lily.github.io//spider)

- **说明**：该数据集是一个人工注释的大规模，复杂且跨领域的文本到SQL数据集。它还是唯一具有多张表格的TB-QA数据集，其中的Schema可以建模为图结构。其中的数据来自六个基础数据集:

	- Restaurants (Tang and Mooney,2001; Popescu et al., 2003b)
	- GeoQuery (Zelle and Mooney, 1996)
	- Scholar (Iyer et al., 2017)
	- Academic (Li and Jagadish, 2014)
	- Yelp (Yaghmazadeh et al., 2017)
	- IMDB (Yaghmazadeh et al., 2017)

	共有11840个问题，6445个复杂SQL查询和206个包含多张表的数据库。

| Method                            | ACC(DEV) | ACC(TEST) | 论文                                                         | 年份 | Code                                          |
| --------------------------------- | -------- | --------- | ------------------------------------------------------------ | ---- | --------------------------------------------- |
| RATSQL + GraPPa (DB content used) | 73.4     | 69.6      | [GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing](https://arxiv.org/pdf/2009.13845.pdf) | 2020 | https://yale-lily.github.io//spider           |
| RYANSQL v2 + BERT                 | 70.6     | 60.6      | [RYANSQL: Recursively Applying Sketch-based Slot Fillings for Complex Text-to-SQL in Cross-Domain Databases](https://arxiv.org/pdf/2004.03125v1.pdf) | 2020 |                                               |
| MASmBoP + BART                    | 64.5     |           | [ TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data](https://arxiv.org/pdf/2005.08314v1.pdf) | 2020 | https://github.com/facebookresearch/tabert    |
| GNN                               | 40.7     | 39.4      | [Representing Schema Structure with Graph Neural Networks for Text-to-SQL Parsing](https://www.aclweb.org/anthology/P19-1448.pdf) | 2019 | https://github.com/benbogin/spider-schema-gnn |

#### TabMCQ

- **相关论文**：Tables as Semi-structured Knowledge for Question Answering
- **下载链接**：[链接](http://ai2-website.s3.amazonaws.com/data/TabMCQ_v_1.0.zip)
- **说明**：该数据集包含9092个手动注释的多项选择题（MCQ）及其答案，需要63个表格内容中查询答案。而这些表格是半结构化表，每个表中的行都是带有定义明确的重复填充模式的句子。

####  Sequential QA

- **相关论文**：Search-based neural structured learning for sequential question answering
- **下载链接**：[链接](https://www.microsoft.com/en-us/download/details.aspx?id=54253)
- **说明**：该数据集根据WikiTableQuestions的表格和答案组成序列，每个序列包含较简单但相互关联的问题 。设计数据集时，每个问题都可以由一个或多个表格单元格回答。它由6066个问题序列组成，总计17553个问题（每个序列平均2.9个问题）。

####  Tabfact

- **相关论文**：Tabfact: A large-scale dataset for table-based fact verification
- **下载链接**：[链接](https://github.com/wenhuchen/Table-Fact-Checking)
- **说明**：该数据集由针对16573个Wikipedia表格的117854条手动注释的语句组成，它们的关系分为ENTAILED和REFUTED。这是第一个对结构化数据进行语言推理的数据集，其中涉及符号和语言方面的混合推理技能。 

## 2. 方法总结

### 2.1 表格检索

表格检索是表格问答的基础环节，其质量直接影响表格问答的结果。这里我们简要介绍该任务常用的特征。在实际系统中，这类特征可以通过特征融合的方式统一使用。

#### 2.1.1 表格候选检索

从表格全集中快速筛选得到一个表格子集![](http://latex.codecogs.com/gif.latex?T=\{T_{1}, ... , T_{n}\})，并尽量保证与Q最相关的表格包含在T之中。

- **表格全集相对有限**：可以将每个表格的结构打散并将内容顺序连接构成一个“文档”，然后基于现有文本检索技术找到与输入问题Q最相关的表格子集T
- **表格全集为互联网所有表格**：需要先基于搜索引擎找到与问题最相关的结果页面集合。然后抽取该结果网页集合中包含的全部表格作为表格子集T。这一过程无法保证排名靠前网页中包含的表格和输入问题之间也存在较高的相关性，这也就引入了下一个任务。

#### 2.1.2 表格候选打分

表格候选打分负责计算表格子集T中每个表格和问题Q的相关度，并选择与Q相关度最高的表格![](http://latex.codecogs.com/gif.latex?T_{best})作为表格检索的结果：

![](http://latex.codecogs.com/gif.latex?T_{best}=argmax_{T_{i}\in{T}}\sum_{i=1}^{N}\lambda_{i}\cdot{h_{i}(Q,T_{i})})

其中![](http://latex.codecogs.com/gif.latex?\{h_{i}(Q,T_{i})\}_{i=1}^{N})表示N个特征函数，每个特征函数用来衡量Q和![](http://latex.codecogs.com/gif.latex?T_{i})的某种相关性。![](http://latex.codecogs.com/gif.latex?\{\lambda_{i}\}_{i=1}^{N})是特征函数集合对应的特征权重集合，它通过在标注数据上使用机器学习算法训练得到。

- 如果将表格看作文档，可以采用文本检索中最常见的BM25作为一个表格候选打分特征。

- 如果将表格看作字符串，忽略表格结构，最长公共子串、编辑距离在内的多种字符串匹配方法都可以用来设计匹配特征。
- [Cafarella等人](https://sirrice.github.io/files/papers/webtables-vldb08.pdf)介绍Google搜索引擎中表格检索模块所使用的部分特征，包括表格的行数，列数，内容为空的表格单元数，表格所在网页在网页搜索结果中的排序，问题和表头的匹配度、问题和表格标题的匹配度、问题和表格单元集合的匹配度、问题和最左边第一列表格单元集合的匹配度等。
- [Balakrishnan等人](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43806.pdf)基于知识图谱对表格单元进行实体匹配，根据表格单元匹配实体的类型标记表格单元和对应表头，并使用该信息设计问题和表格之间的相关度特征。
- [Yan等人](https://arxiv.org/pdf/1706.02427.pdf)提出基于神经网络的表格检索模型，通过将输入问题和表格转化为向量表示，计算二者之间的相似度。

### 2.2 答案生成

#### 2.2.1 基于答案排序的方法

基于答案排序的方法假设输入问题Q对应的答案A一定存在于给定表格T中，因此可以通过对表格中不同表格单元进行打分和排序，选择问题对应的答案。

#### 2.2.2 基于语义分析的方法

基于语义分析的方法通过两步完成任务。第一步，基于表格T对问题Q进行语义分析，将其转化为对应的语义表示LF；第二步，将LF作为结构化查询，通过在T上执行得到问题对应的答案。和KBQA类似，基于表格的语义分析方法既可以基于<问题，答案>标注间接训练语义分析模型，也可以基于<问题，语义表示>标注直接训练语义分析模型。

#### 2.2.3 基于神经网络的方法

该类方法训练端到端的神经网络模型，直接生成问题对应的答案。与前面两类方法相比，基于神经网络的方法不需要人工设计特征，也不需要对问题生成显示的语义表示。因此，这类方法只需要带有答案标注的表格问答数据集作为训练数据，着能够极大减少训练数据所需要的标注成本。

## 3. Paper List

| 会议/年份  | 论文                                                         | 链接                                                         |
| :--------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| AAAI2020   | CFGNN: Cross Flow Graph Neural Networks for Question Answering on Complex Tables | https://aaai.org/ojs/index.php/AAAI/article/view/6506        |
| ACL2020    | TaPas: Weakly Supervised Table Parsing via Pre-training      | https://arxiv.org/pdf/2004.02349.pdf                         |
| ACL2019    | Representing Schema Structure with Graph Neural Networks for Text-to-SQL Parsing | https://www.aclweb.org/anthology/P19-1448.pdf                |
| NAACL2019  | Iterative Search for Weakly Supervised Semantic Parsing      | https://www.aclweb.org/anthology/N19-1273.pdf                |
| EMNLP2019  | Clause-wise and recursive decoding for complex and cross-domain text-to-SQL generation | https://www.aclweb.org/anthology/D19-1624.pdf                |
| EMNLP2019  | Answering Conversational Questions on Structured Data without Logical Forms | https://www.aclweb.org/anthology/D19-1603.pdf                |
| ICML2019   | Learning to Generalize from Sparse and Underspecified Rewards | https://arxiv.org/pdf/1902.07198.pdf                         |
| ACL2018    | Semantic Parsing with Syntax- and Table-Aware SQL Generation | https://www.aclweb.org/anthology/P18-1034.pdf                |
| NAACL2018  | TypeSQL: Knowledge-based type-aware neural text-to-SQL generation | https://www.aclweb.org/anthology/N18-2093.pdf                |
| EMNLP2018  | SyntaxSQLNet: Syntax Tree Networks for Complex and Cross-Domain Text-to-SQL Task | https://www.aclweb.org/anthology/D18-1193.pdf                |
| COLING2018 | A Neural Question Answering Model Based on Semi-Structured Tables | https://www.aclweb.org/anthology/C18-1165.pdf                |
| ECIR2018   | Neural Multi-step Reasoning for Question Answering on Semi-structured Tables | https://link.springer.com/chapter/10.1007%2F978-3-319-76941-7_52 |
| ACL2017    | Search-based Neural Structured Learning for Sequential Question Answering | https://www.aclweb.org/anthology/P17-1167.pdf                |
| ACL2017    | Learning a Neural Semantic Parser from User Feedback         | https://www.aclweb.org/anthology/P17-1089.pdf                |
| EMNLP2017  | Macro Grammars and Holistic Triggering for Efficient Semantic Parsing | https://www.aclweb.org/anthology/D17-1125.pdf                |
| EMNLP2017  | Neural semantic parsing with type constraints for semi-structured tables | https://www.aclweb.org/anthology/D17-1160.pdf                |
| ICLR2017   | Learning a natural language interface with neural programmer | https://openreview.net/pdf?id=ry2YOrcge                      |
| ACL2016    | Tables as Semi-structured Knowledge for Question Answering   | https://www.aclweb.org/anthology/P16-1045.pdf                |
| WWW2016    | Table Cell Search for Question Answering                     | https://dl.acm.org/doi/pdf/10.1145/2872427.2883080           |

## 4. 相关资料

[表格问答1：简介](https://zhuanlan.zhihu.com/p/128123561)

[表格问答2：模型](https://mp.weixin.qq.com/s?__biz=MzAxMDk0OTI3Ng==&mid=2247484103&idx=1&sn=73f37fbc1dbd5fdc2d4ad54f58693ef3&chksm=9b49c534ac3e4c222f6a320674b3728cf8567b9a16e6d66b8fdcf06703b05a16a9c9ed9d79a3&scene=21#wechat_redirect)

[你已经是个成熟的表格了，该学会自然语言处理了](https://www.msra.cn/zh-cn/news/features/table-intelligence)

[机器阅读理解 | (1) 智能问答概述](https://blog.csdn.net/sdu_hao/article/details/104172875#2.%20%E8%A1%A8%E6%A0%BC%E9%97%AE%E7%AD%94)

[《智能问答》 高等教育出版社 段楠 周明著](https://www.msra.cn/zh-cn/news/features/book-recommendation-qa-mt)
