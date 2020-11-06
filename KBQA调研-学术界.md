# KBQA调研——学术界

## 目录

  * [1. 任务](#1-任务)
     * [1.1. 背景](#11-背景)
     * [1.2. 任务定义](#12-任务定义)
     * [1.3. 数据集](#13-数据集)
     * [1.4. SOTA](#14-SOTA)
     * [1.5. 评测标准](#15-评测标准)
     
  * [2. 方法总结](#2-方法总结)
     * [2.1. 基于语义解析（Semantic Parsing）的方法](#21-基于语义解析semantic-parsing的方法)
     * [2.2. 基于信息抽取（Information Extraction）的方法](#22-基于信息抽取information-extraction的方法)
        * [2.2.1. 候选答案的得出](#221-候选答案的得出)
        * [2.2.2. 问题的信息抽取](#222-问题的信息抽取)
        * [2.2.3. 训练分类器，判断候选答案是否正确](#223-训练分类器判断候选答案是否正确)
     * [2.3. 基于向量建模（Vector Modeling）的方法](#23-基于向量建模vector-modeling的方法)
  * [3. Paper List](#3-paper-list)
     * [3.1. 论文列表](#31-论文列表)
     * [3.2. 论文解读](#32-论文解读)
  * [4.相关链接](#4-相关链接)
  * [5.参考资源](#4-参考资源)

## 1. 任务

### 1.1. 背景
（1）Knowledge Base
KB中包括三类元素：实体（entity）、关系（relation），以及属性（literal）。实体代表一些人或事物，关系用于连接两个实体，表征它们之间的一些联系，如实体Michael Crichton与实体Chicago之间就可以由关系bornin连接，代表作家Michael Crichton出生于城市Chicago。同时，关系不仅可以用于连接两个实体，也可以连接实体和某属性，如关系area可用于连接Chicago和属性606km2，表明chicago面积为606km2。
用更形式化的语言来描述：KB可以表示为三元组的集合，三元组为（entity，relation，entity/literal）。


（2）Formal Query Languages
SPARQL，λ-DCS、FunQL等查询语言可以用于查询以及操作KG中存储的数据。这些语言具有明确定义的形式语法和结构，并允许进行复杂的检索。 SPARQL是KB最常用的查询语言之一，DBpedia和Freebase等许多公共可用KB都支持SPARQL。


（3）question answering over KB
给定自然语言问题（NLQ），对问题进行理解和解析，利用KB得到正确答案。 

### 1.2. 任务定义
知识库问答（knowledge based question answering,KB-QA）：给定自然语言形式的问题，通过对问题进行语义理解和解析，进而利用知识库进行查询、推理，最终得出答案。
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/kbqa%20definition.png)

（注：该图来自中科院刘康老师的报告）

### 1.3. 数据集
- FREE917:第一个大规模的KBQA数据集，于2013年提出，包含917 个问题，同时提供相应逻辑查询，覆盖600多种freebase上的关系。
- Webquestions：数据集中有6642个问题答案对，数据集规模虽然较FREE917提高了不少，但有两个突出的缺陷：没有提供对应的查询，不利于基于逻辑表达式模型的训练；另外webquestions中简单问句多而复杂问句少。
- WebQSP：是WEBQUESTIONS的子集，问题都是需要多跳才能回答，属于multi-relation KBQA dataset，另外补全了对应的查询句。
- Complexquestion、GRAPHQUESTIONS：在问句的结构和表达多样性等方面进一步增强了WEBQUESTIONSP，，包括类型约束，显\隐式的时间约束，聚合操作。
- SimpleQuestions：数据规模较大，共100K，数据形式为(quesition，knowledge base fact)，均为简单问题，只需KB中的一个三元组即可回答,即single-relation dataset。
- FACTOID QUESTIONS：将SimpleQuestion扩展为含30M句的FACTOID QUESTIONS，只包含答案不含问句。
- QALD-6：QALD有几个子任务，QALD-6是英语的QA任务，目标KB是DBpedia。训练集350个问题，测试集100个问题，提供 SPARQL查询和问题相应答案集。虽然数据集规模较小，但是更为口语化、复杂。
- QALD-9：2018年发布，是QALD1-QALD8的超集。
- LC-QuAD：包含5000对问题及其相应的SPARQL查询的问答数据集。目标知识库是DBpedia-April。
- LC-QuAD2：发布了大规模的数据集LC-QuAD2，包含30000个问题，同时也提供相应的SPARQL查询。
- MetaQA：基于MovieQA的电影KBQA数据集，数据集中已将问题按跳数进行了区分，其中1跳116045个问题答案对，2跳148724组问题答案对，3跳142744个问题答案对。
- PQ：采用Freebase的两个子集和模板来构造的数据集，通过搜索互联网和两个现实世界的数据集WebQuestions和WikiAnswers，为关系提供了解释模板和同义词，提高了语言的多样性。
- PQL：使用更大的KB来构造数据，并且相对于PQ提供更少的训练集，整体难度高于PQ。

| 数据集              | 数据规模                                                | 地址                                                         |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Free917              |917         |https://github.com/pks/rebol/tree/master/data/free917         |
| WebQuestions        | 5810  | https://github.com/brmson/dataset-factoid-webquestions       |
| WebQuestionsSP      | 4737 | https://www.microsoft.com/en-us/download/details.aspx?id=52763 |
| ComplexQuestions | 2100 | https://github.com/JunweiBao/MulCQA/tree/ComplexQuestions |
| ComplexWebQuestions | 34689                       | https://www.tau-nlp.org/compwebq                            |
| SimpleQuestions     | 100K                  | https://research.fb.com/downloads/babi/                      |
| FACTOID QUESTIONS   | 30M |  http://academictorrents.com/details/973fb709bdb9db6066213bbc5529482a190098ce|
| GraphQuestions      | 5166                | https://github.com/ysu1989/GraphQuestions                    |
| LC-QuAD             | 5000                      | https://github.com/AskNowQA/LC-QuAD                          |
| LC-QuAD 2.0         | 30000                                | http://lc-quad.sda.tech/                                     |
| QALD-6              | 450          | https://github.com/ag-sc/QALD/tree/master/6/data             |
| QALD-9              | 350               | https://github.com/ag-sc/QALD/tree/master/9                  |
| MetaQA | 407513 | https://github.com/yuyuz/MetaQA |
| PQ | 7106 | https://github.com/zmtkeke/IRN |
| PQL | 2625 | https://github.com/zmtkeke/IRN |

### 1.4. SOTA(leaderboard)



WebQuestions:
| 模型              | average F1                                        |论文题目|年份|论文链接|code|
| ----------------- | ------------------------------------------|--|--|--|--|
|APVA-TURBO|63.4|The APVA-TURBO Approach To Question Answering in Knowledge Base|2018|https://www.aclweb.org/anthology/C18-1170.pdf||
|STF|53.6%|A State-transition Framework to Answer Complex Questions over Knowledge Base|2018|https://www.aclweb.org/anthology/D18-1234.pdf||
|STAGG|52.5%|Semantic Parsing via Staged Query Graph Generation:Question Answering with Knowledge Base|2015|https://www.aclweb.org/anthology/P15-1128.pdf|https://github.com/scottyih/STAGG|
|QUINT|51.0%|Automated Template Generation for Question Answering over Knowledge Graphs|2017|http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p1191.pdf||
|NFF|49.6%|Answering natural language questions by subgraph matching over knowledge graphs|2017|https://ieeexplore.ieee.org/document/8085196|https://github.com/pkumod/gAnswer|
|Aqqu|49.4%|More Accurate Question Answering on Freebase|2015|http://ad-publications.informatik.uni-freiburg.de/freebase-qa.pdf|https://github.com/ad-freiburg/aqqu|



ComplexWebQuestions:
| 模型              | P@1                                        |论文题目|年份|论文链接|code|
| ------------------- | -----------------------------------------|--|--|--|--|
|PullNet|45.90|PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text|2019|https://www.aclweb.org/anthology/D19-1242.pdf||
|SPLITQA + data augmentation|34.20|Repartitioning of the COMPLEXWEBQUESTIONS Dataset|2018|https://arxiv.org/pdf/1807.09623.pdf||
|SPARQA|31.57|SPARQA: Skeleton-based Semantic Parsing for Complex Questions over Knowledge Bases|2020|https://arxiv.org/pdf/2003.13956.pdf|https://github.com/nju-websoft/SPARQA|
|MHQA-GRN|30.10|Exploring Graph-structured Passage Representation for Multi-hop Reading Comprehension with Graph Neural Networks|2018|https://arxiv.org/pdf/1809.02040.pdf||
|SPLITQA + PRETRAINED|25.90|Repartitioning of the COMPLEXWEBQUESTIONS Dataset|2018|https://arxiv.org/pdf/1807.09623.pdf||
|SIMPQA + PRETRAINED|19.90|Repartitioning of the COMPLEXWEBQUESTIONS Dataset|2018|https://arxiv.org/pdf/1807.09623.pdf||


SimpleQuestions：
| 模型              |ACC                                        |论文题目|年份|论文链接|code|
| ------------------- | -----------------------------------------|--|--|--|--|
|MVA-MTQA-net(MTL)|95.7|Multi-Task Learning for Conversational Question Answering over a Large-Scale Knowledge Base|2019|https://www.aclweb.org/anthology/D19-1248.pdf|https://github.com/taoshen58/MaSP|
|AR-SMCNN|93.7|Question Answering over Freebase via Attentive RNN with Similarity Matrix based CNN|2018|https://arxiv.org/vc/arxiv/papers/1804/1804.03317v2.pdf|https://github.com/quyingqi/kbqa-ar-smcnn|
|HR-BiLSTM|93.3|Improved Neural Relation Detection for Knowledge Base Question Answering|2017|https://www.aclweb.org/anthology/P17-1053.pdf|https://github.com/StevenWD/HR-BiLSTM|
|ComplexQueryGraphs|93.1|Knowledge Base Question Answering via Encoding of Complex Query Graphs|2018|https://www.aclweb.org/anthology/D18-1242.pdf|https://github.com/FengliLin/EMNLP2018-KBQA|
|AMPCNN|91.3|Simple Question Answering by Attentive Convolutional Neural Network|2016|https://www.aclweb.org/anthology/C16-1164.pdf||
|STAGG|90.0|Semantic Parsing via Staged Query Graph Generation:Question Answering with Knowledge Base| 2015|https://www.aclweb.org/anthology/P15-1128.pdf|https://github.com/scottyih/STAGG|





GraphQuestion:
| 模型              | F1                                        |论文题目|年份|论文链接|code|
| ------------------- | -----------------------------------------|--|--|--|--|
|SPARQA|21.53|SPARQA: Skeleton-based Semantic Parsing for Complex Questions over Knowledge Bases|2020|https://arxiv.org/pdf/2003.13956.pdf|https://arxiv.org/pdf/2003.13956.pdf|https://github.com/nju-websoft/SPARQA|
|PARA4QA|20.40|Learning to Paraphrase for Question Answering|2017|https://arxiv.org/pdf/1708.06022.pdf|-|
|UDEPLAMBDA|17.70|Universal Semantic Parsing|2017|https://arxiv.org/pdf/1702.03196.pdf|https://github.com/sivareddyg/udeplambda|
|SCANNER|17.02|Learning Structured Natural Language Representations for Semantic Parsing|2017|https://arxiv.org/pdf/1704.08387.pdf||
|PARASEMPRE|12.79|Semantic Parsing via Paraphrasing|2014|https://www.aclweb.org/anthology/P14-1133.pdf|-|
|SEMPRE|10.80|Semantic Parsing on Freebase from Question-Answer Pairs|2013|https://www.aclweb.org/anthology/D13-1160.pdf|https://github.com/percyliang/sempre|
|JACANA|5.08|2014|Information Extraction over Structured Data: Question Answering with Freebase|http://cs.jhu.edu/~xuchen/paper/yao-jacana-freebase-acl2014.pdf|https://github.com/xuchen/jacana|




ComplexQuestions:
| 模型              | average F1                                        |论文题目|年份|论文链接|code|
| ------------------- | -----------------------------------------|--|--|--|--|
|STF|54.3%|A State-transition Framework to Answer Complex Questions over Knowledge Base|2018|https://www.aclweb.org/anthology/D18-1234.pdf||
|QUINT|49.2%|Automated Template Generation for Question Answering over Knowledge Graphs|2017|http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p1191.pdf||
|Aqqu++|46.7%|More Accurate Question Answering on Freebase|2015|http://ad-publications.informatik.uni-freiburg.de/freebase-qa.pdf|https://github.com/ad-freiburg/aqqu|
|Aqqu|27.8%|More Accurate Question Answering on Freebase|2015|http://ad-publications.informatik.uni-freiburg.de/freebase-qa.pdf|https://github.com/ad-freiburg/aqqu|




QALD-6:
| 模型              | F1                                        |论文题目|年份|论文链接|code|
| ----------------- | ------------------------------------------|--|--|--|--|
|CaNaLi|0.89|Answering Controlled Natural Language Questions on RDF Knowledge Bases|2016|https://openproceedings.org/2016/conf/edbt/paper-259.pdf||
|STF|0.80|A State-transition Framework to Answer Complex Questions over Knowledge Base|2018|https://www.aclweb.org/anthology/D18-1234.pdf||
|NFF|0.78|Answering natural language questions by subgraph matching over knowledge graphs|2017|https://ieeexplore.ieee.org/document/8085196|https://github.com/pkumod/gAnswer|
|UTQA|0.75|||||
|KWGAnswer|0.70||||
|gAnswer|0.55|Natural language question answering over RDF - A graph data driven approach|2014|https://www.researchgate.net/publication/266656635_Natural_language_question_answering_over_RDF_-_A_graph_data_driven_approach||
|Aqqu|0.38|More Accurate Question Answering on Freebase|2015|http://ad-publications.informatik.uni-freiburg.de/freebase-qa.pdf|https://github.com/ad-freiburg/aqqu|
|SemGraphQA|0.37|SemGraphQA@QALD-5: LIMSI participation at QALD-5@CLEF|2015|https://pdfs.semanticscholar.org/59e5/b01f7a634218cace37c47484073bbdd25138.pdf|-|




### 1.5. 评测标准

- Accuracy：当预测答案属于提供的问题答案之一时就算正确。
- average f1
- Hits@1
- P@1:Precision@1

## 2. 方法总结
可以划分为三类：基于语义解析（Semantic Parsing）的方法，基于信息抽取（Information Extraction）的方法，基于向量建模（Vector Modeling）的方法。

### 2.1. 基于语义解析（Semantic Parsing）的方法

将自然语言转化为一系列形式化的逻辑形式（logic form）,通过对逻辑形式进行自底向上的解析，得到一种可以表达整个问题语义的逻辑形式，通过相应的查询语句在知识库中进行查询，从而得出答案。

语义解析的一个经典baseline方法，来自《Semantic Parsing on Freebase from Question-Answer Pairs》

语义解析最关键的环节就是将自然语言形式的问题转换为逻辑形式，逻辑形式之后可以通过特定的逻辑语言对knowledge base进行查询。该文中的方法是叶节点自底向上构造语法树，最终得到的树根节点就是自然语言问题对应的逻辑形式。语法树的构建分为两个步骤：
- 第一步：建立词汇表，将句子中的词映射为KB中的实体或关系。实体采用一些字符串匹配方式进行映射。复杂的部分是将动词短语如“was also born in”，映射到相应的知识库实体关系上，如PlaceOfBirth， 则较难通过字符串匹配的方式建立映射。作者是使用统计的方法来做：在文档中，如果有较多的实体对（entity1，entity2）作为主语和宾语出现在was also born in的两侧，并且，在知识库中，这些实体对也同时出现在包含PlaceOfBirth的三元组中，那么我们可以认为“was also born in”这个短语可以和PlaceOfBirth建立映射。值得注意的是由于自然语言短语和知识库实体关系的对应关系是多对多的，比如“was also born in”可能对应PlaceOfBirth，也可能对应DateOfBrith，需要对此进行区分。作者使用entity1、entity2的类别来区分这种对应关系。
- 第二步：经过第一步得到语法树的叶节点之后，自上而下构建语法树，文章中对任意两个叶节点都进行了逻辑形式的所有可以进行的操作（join，intersection，aggregation），得到了所有可能的语法树。

如下图片为自然语言问题“where was Oboma born?”转换为逻辑形式的过程：
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/%E8%AF%AD%E4%B9%89%E8%A7%A3%E6%9E%90%E7%94%9F%E6%88%90%E9%80%BB%E8%BE%91%E5%BD%A2%E5%BC%8F.PNG)

经过以上两步可以获得候选语法树，之后训练分类器，求出自然语言问题在代表不同逻辑形式的候选语法树上的概率分布。至此完成了语义解析方法中最重要的步骤。

### 2.2. 基于信息抽取（Information Extraction）的方法
提取问题中的实体，通过在知识库中查询该实体可以得到以该实体节点为中心的知识库子图，子图中的每一个节点或边都可以作为候选答案，通过观察问题依据某些规则或模板进行信息抽取，得到表征问题和候选答案特征的特征向量，建立分类器通过输入特征向量对候选答案进行筛选，从而得出最终答案。

一个经典的信息抽取baseline方法，来自《Information Extraction over Structured Data: Question Answering with Freebase》，主要介绍该文中如何得出候选答案，如何对问题进行信息抽取，以及最终判断候选答案是否为正确答案。

#### 2.2.1. 候选答案的得出
使用命名实体识别来确定问题的主题词，之后entity linking到KB，选择该实体2 hop以内关系和实体，形成知识库的一个子图，该图中的点与边都是候选答案。

#### 2.2.2. 问题的信息抽取
将问题的dependency tree转换为question graph，主要操作有：提取问题词qword（how，why，when之类的词），问题焦点qfocus（time，place等），问题主题词qtopic和问题中心动词qverb这四个问题特征，将这些词语在dependency tree上做标注，同时删去dependency tree上不重要的节点（如冠词，标点）。经过这一转换过程，可以找到问题中最关键的要素，完成了对问题的信息抽取。

如下图片中是一个将dependency tree转换为question graph的例子：
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/%E5%B0%86%E4%BE%9D%E5%AD%98%E5%85%B3%E7%B3%BB%E6%A0%91%E8%BD%AC%E6%8D%A2%E4%B8%BA%E9%97%AE%E9%A2%98%E5%9B%BE.PNG)

#### 2.2.3. 训练分类器，判断候选答案是否正确
分类器的输入特征是问题和某一个候选答案的特征结合形成。

问题图中每一条边e(s,t)，可以抽取4种问题特征：s，t，s|t，和s|e|t。

候选答案的特征是该节点在KB中的所有属性和关系，这些属性和关系可以反映出候选答案实体的特征。

得到问题和候选答案的特征之后，可以由此组成分类器的输入特征。每一个问题-候选答案特征由问题特征中的一个特征和候选答案特征中的一个特征组合而成，这样分类器就可以得到很多特征用于判别该候选答案是否为正确答案。

### 2.3. 基于向量建模（Vector Modeling）的方法
向量建模的方法与信息抽取有相似之处，前期操作都是通过把问题中的主题词映射到kb的实体，得到候选答案。基于向量建模的方法把问题和候选答案分别映射到低维空间，得到它们的分布式表达（Distributed Embedding），通过训练数据对该分布式表达进行训练，使得问题向量和它对应的正确答案向量在低维空间的关联得分（通常以点乘为形式计算）尽量高。

下面介绍一个向量建模的经典方法，来自《Question answering with subgraph embeddings》。

向量建模方法的核心步骤是将问题和候选答案分别映射到低维空间，得到它们的分布式表达。本文的方法如下：
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/%E5%90%91%E9%87%8F%E5%BB%BA%E6%A8%A1.PNG)

**问题的分布式表达：**

问题维度为字典大小+KB中实体个数+KB中关系个数（加后面的两个应该是为了之后点乘计算方便）,类似词袋模型，根据一个问题中这些词语是否出现，在相应位置分别置0或1。

**答案的分布式表达：**

因为候选答案只有一个，如果采用和上述问题的相同方法，则答案的表示是one-hot的，由于没有引入KB中的知识， 就将KBQA问题退化成QA问题了。因此同时问题的表达中添加：从主题词词到到候选答案的路径，同时将候选答案在KB中的属性和关系也表示在分布式表达中。

经过上面两步的处理可以得到问题和候选答案的multi-hot表达，采用类似word embedding的方法训练，可以得到问题和候选答案的dirtributed embedding。基本思路为：将Multi-hot的输入通过矩阵映射，可以得到目标维数的分布式表达，这个矩阵就是要训练的参数。

得到问题和答案的分布式表达之后，采用点乘的方式对候选答案打分。

## 3. Paper List
### 3.1. 论文列表

| 会议/年份  | 论文 |链接|
| ------------- | ------------- |------------- |
| AAAI2020  |  skeleton-based Semantic Parsing for Complex Questions over Knowledge Bases |https://arxiv.org/pdf/2003.13956.pdf|
| AAAI2019  |  Multi-Task Learning with Multi-View Attention for Answer Selection and Knowledge Base Question Answering |https://arxiv.org/pdf/1812.02354.pdf|
| AAAI2018  | variational reasoning for question answering with knowledge graph  |https://arxiv.org/pdf/1709.04071.pdf|
| EMNLP2019 | Multi-Task Learning for Conversational Question Answering over a Large-Scale Knowledge Base  |https://www.aclweb.org/anthology/D19-1248.pdf|
| EMNLP2019 | PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text |https://www.aclweb.org/anthology/D19-1242.pdf|
| EMNLP2018  |  A State-transition Framework to Answer Complex Questions over Knowledge Base |https://www.aclweb.org/anthology/D18-1234.pdf|
| EMNLP2018  | knowledge base question answering via encoding of complex query graphs  |https://www.aclweb.org/anthology/D18-1242.pdf|
| EMNLP2018 |  Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text |https://arxiv.org/pdf/1809.00782.pdf|
| EMNLP2017  |  QUINT:Interpretable Question Answering over Knowledge Bases |https://www.aclweb.org/anthology/D17-2011.pdf|
| ACL2020 | Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases |https://www.aclweb.org/anthology/2020.acl-main.91.pdf|
| ACL2020 | Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings |https://www.aclweb.org/anthology/2020.acl-main.412.pdf|
| ACL2019 | Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader |https://www.aclweb.org/anthology/P19-1417.pdf|
| ACL2017 | Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks  |https://arxiv.org/pdf/1704.08384.pdf|
| CIKM2019 |  Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader |https://www.aclweb.org/anthology/P19-1417.pdf|
| CIKM2019 | Learning to Answer Complex Questionsover Knowledge Bases with Query Composition |https://www.researchgate.net/publication/337017270_Learning_to_Answer_Complex_Questions_over_Knowledge_Bases_with_Query_Composition|
| CIKM2019  |  Message Passing for Complex Question Answering over Knowledge Graphs |https://arxiv.org/pdf/1908.06917.pdf|
| IJCAI2020 | Two-Phase Hypergraph Based Reasoning with Dynamic Relations for Multi-Hop KBQA. |https://www.ijcai.org/Proceedings/2020/0500.pdf|
| IJCAI2020 | Formal Query Building with Query Structure Prediction for Complex Question Answering over Knowledge Base |https://www.ijcai.org/Proceedings/2020/0519.pdf|
| IJCAI2019  |  Neural Program Induction for KBQA Without Gold Programs or Query Annotations |https://www.ijcai.org/Proceedings/2019/0679.pdf|
| IJCAI2019|  Knowledge Base Question Answering with Topic Units |https://www.ijcai.org/Proceedings/2019/0701.pdf|
| NAACL2019 | Enhancing Key-Value Memory Neural Networks for Knowledge Based Question Answering |https://www.aclweb.org/anthology/N19-1301.pdf|
| NAACL2019 | UHop: An Unrestricted-Hop Relation Extraction Framework for Knowledge-Based Question Answering |https://arxiv.org/pdf/1904.01246.pdf|
|NAACL2019|Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases|https://arxiv.org/pdf/1903.02188.pdf|
|COLING2018|Modeling Semantics with Gated Graph Neural Networks for Knowledge Base Question Answering|https://www.aclweb.org/anthology/C18-1280.pdf|
|COLING2018|The APVA-TURBO Approach To Question Answering in Knowledge Base|https://www.aclweb.org/anthology/C18-1170.pdf|


### 3.2. 论文解读

>《skeleton-based Semantic Parsing for Complex Questions over Knowledge Bases》

这篇论文指出了semantic parsing的两大问题：

1. semantic parsing 模型一般都会基于依存语法分析，导致依存分析的错误会传递到下游模型上。并且依存分析在复杂的长句子上很容易出错。
2. 通常会把问题转化为一个查询语句，然后在数据库中执行。但生成的查询语句往往存在和数据库不一致的情况，包括关系指称的不一致和结构的不一致，会导致找不到答案的问题。

为了解决这两个问题，这篇文章提出了SPARQA模型

1. 提出了用来表示复杂问句的高级表示方法skeleton grammar
2. 在两个KBQA数据集上标注了skeleton
3. 提出评估ungrounded查询的模型

这篇文章使用skeleton辅助semantic parsing，避免依存分析工具的错误，是一种创新的思路。

> 《Multi-Task Learning with Multi-View Attention for Answer Selection and Knowledge Base Question Answering》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/multi%20task%20QA%20netword.PNG)

answer selection和KBQA是QA中两项重要的任务，现有的方法一般都将二者分开来做。作者认为这两项任务有着内在的关联，如：二者在本质上都是ranking的问题，AS任务可以获得KB中的先验知识，KBQA也能通过AS得到信息，因此这两项任务可以从对方得到有用的信息。论文中提出了一种multi-task的学习方法，首先在task-specific层对两个任务的输入单独做encode（word sequence使用BiLSTM做encoder，knowledge sequence由于是离散的信息，因此采用CNN做encoder）。经过task-specific层对两个任务的独立编码之后，在shared层结合两个AS、KBQA两个任务的representation，使用的神经网络是BiLSTM。shared层的输出最终使用KBQA softmax、AS softmax分别得到这两个任务的结果。
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/multi%20view%20attention.PNG)

为了在表示空间增强不同任务之间的相互作用，作者提出一种multi-view的注意力机制，不仅仅利用task-specific层的attention，还结合了shared层的attention。其次，从word-level和knowledge-level这两个视角分别获得注意力信息。具体来说，有5个视角的attention：word,knowledge, semantic, knowledge semantic and co-attention。
word view attention由question和answer的word sequence计算得到，计算方式为Mw = tanh(Ewq*Uw*Ewa),得到一个矩阵，然后对行、列分别进行max-pooling，可以得到word view的问题和答案注意力权重，其他view的注意力权重计算方式类似。
knowledge view attention 由question和answer的knowledge sequence计算得到。
semantic view是对task-specific层的输出做max/average pooling，从而得到语义信息。
co-attention view：得到最终的question和answer之间attention，计算方式与word view类似。
结合以上几种attention，得到最终的attention权重，shared层输出的question和answer与权重相乘就可以得到最后的问题和答案representation，因为考虑了很多层面的信息，因此这一表达应当信息很丰富。

> 《variational reasoning for question answering with knowledge graph》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/VRN.PNG)

文章首先提出了现有的基于语义解析方法的问题：（1）知识库中需要多跳才能获取的答案则无法回答（2）在实际应用中输入的问题通常存在噪声，此时语义解析就很难通过类似字符串匹配的方法找到句子中的topic entity。
为了解决上述问题，模型分为两部分：第一部分是通过概率模型来识别问句中的实体。第二部分则是在问答时在知识图谱上做逻辑推理，在推理这部分的工作中我们给出了上一步识别的实体和问句希望系统能给出答案。文章中提出的模型在现在的数据集上跑出了较好的结果，为了验证在存在噪声数据上的效果，作者还在以人声为输入的数据上进行了实验。

> 《Multi-Task Learning for Conversational Question Answering
over a Large-Scale Knowledge Base》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/multi%20task.PNG)

基于语义解析（semantic parsing）的kbqa方法通常将任务分解为几个子任务并依次解决，这样的方法有着明显的不足：子任务间的错误传递、共享信息困难。本文提出了一种multi task的学习框架：Multi-task Semantic Parsing (MaSP) model，模型由四部分组成：word embedding, contextual encoder, entity detection以及 pointer-equipped logical form decoder.内置的pointer network可以很好地结合到上游实体检测任务。

>PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text
这篇论文的工作是基于GRAFT-Net，为了改进GRAFT-Net启发式子图抽取产生的结果过大并且有时不包含正确答案的问题。PullNet训练了另一个GRAFT-Net模型来完成子图抽取任务。通过训练迭代式的子图构建模型，在保证召回率的同时又缩小了子图规模。

> 《A State-transition Framework to Answer Complex Questions over Knowledge Base》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/SQG%20%20generation.PNG)

该论文主要针对复杂的问题，提出了一个状态转移框架和四种转移操作，可以将自然语言问题转化为语义查询图(semantic query graph (SQG) )从而能够使用现有的查询算法找到答案。与现有工作相比，本文的方法不依赖于人工定义的模板，针对复杂问题能够灵活的生成查询图，在DBpedia和Freebase知识库上多个QA数据集取得了较好的结果。

> 《knowledge base question answering via encoding of complex query graphs》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/%E9%97%AE%E9%A2%98%E5%92%8C%E5%9B%BE%E5%8C%B9%E9%85%8D.PNG)

这篇论文同样主要关注复杂问题的回答，提出了一种基于向量的KBQA方法，将复杂的query structure编码为统一的向量，可以捕获到复杂的问题中不同semantic componentd的关系。首先通过分阶段的生成方法来生成候选图，之后通过神经网络来衡量问题与每个查询图之间的语义相似性，使得问题与最相符的查询图匹配。

> 《Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/early%20fusion.PNG)

在QA任务中，现存的模型主要从外部的知识库（knowledge base）或者从非结构化的文本中寻找答案，也有人用一些方法将来自两个信息源的预测结果进行聚合，本文称之为后期融合，而本文关注的重点是早期融合，将与问题相关的KB实体和文本放在一起，然后训练单个模型提取问题相对应的答案。作者提出了GRAFT-Net (Graphs of Relations Among Facts and Text Networks),将KB实体和文本放入同一个子图，然后训练单个模型从子图中提取答案。

> 《QUINT:Interpretable Question Answering over Knowledge Bases》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/quint.PNG)

提出了一种“QUINT”系统，最大的特点是论文题目中的“可解释性”，具体来说，当QUINT回答问题时，它将可视化从自然语言问题到最终答案的完整推导序列，具有较好的可解释性。QUINT系统的核心在于“role-aligned“模板，通过利用问题和答案，通过自动生成模板的方法，把问题映射成一个查询模板用于查询。

> 《Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/ACL2017.png)

算法分两个部分：
子图阅读器(SGReader)运用图注意力技术获取子图实体的邻居只是，考虑(1)该邻居关系是否与问题相关(2)该邻居实体是否在问题中被提及。经过计算后，SGReader最终输出结合邻居知识的所有相关实体向量e'
文本阅读器(KAReader)：根据已获取的知识信息重构问题，结合问题向量定位文档并聚合相关实体信息获得ed，最终concatenate起来对可能成为问题答案的实体进行预测。

> 《Question Answering on Knowledge Bases and Text using Universal Schema and Memory Networks》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/ACL2019.png)

传统QA要么只依赖KB用，要么只依赖文本，两者结合起来可以从结构化知识库和非结构化文本结合推理答案。本文通过联合嵌入KB和text的facts形成统一的结构化表示，允许信息的交错传播。该universal矩阵每一行都是一个是梯队，每一列代表它们在KB中的关系/文本之间的模式。
文章中使用的数据集是SPADES，包含完形填空式的问题，有93K句子和1.8M实体，KB知识库为freebase，文本材料在ClueWeb。

> 《Learning to Rank Query Graphs for Complex Question Answering over Knowledge Graphs》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/CIKM2019.png)

首先建立查询图，查询图是有向无环图，由 (grounded entity, existential variable, lambda variable, auxiliary function) 组成，其中 grounded entity 是可以链接到KB的实体，中间连接Z者KB中关系，lambda代表了答案实体，existential variable用来disambiguate，auxiliary function是对答案实体的constraints，有答案实体类别，问题类别(ask[是否含有], count, set)
core chain是linear的查询图的subset，不包含constraints，只针对linking到的实体最多两跳构建core chain，其中对predicate的方向用 +/-说明
core chain candidate ranking: 用5中方法进行建模，其中 slot matching model 效果最好
Predicting Auxiliary Constraints
    BiLSTM classify (count, ask, set)
    pair-wise ranking 把正类别和负类别分别编码算相似度
    
> 《Message Passing for Complex Question Answering over Knowledge Graphs》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/CIKM2019_2.png)

问句分析：将回答分为问句解析以及答案推理，其中问句解析把问题定义为q=\<tq,Seqq\>，解析过程为根据问题识别问句类型tq以及n跳序列Seqq=(<EI,Pi,Ci>)hi=1(CRF=BiLSTM)，之后将n跳序列分别和知识库中实体关系匹配(BM25.embedding)。
答案推理：对于每一跳中E,P,C抽出子图，然后通过相邻实体和关系的置信度来计算答案实体的置信度，取得答案。
该模型在LC-QuAD数据集上达到了SOTA。

> 《Neural Program Induction for KBQA Without Gold Programs or Query Annotations》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/IJCAI2019.png)

dataset： WebQuestionsSP Complex Sequential QA
gold input:at par with hand-crafted rule-based models
in the noisy settings >> state-of-the-art models by a significant margin

> 《Knowledge Base Question Answering with Topic Units》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/image/IJCAI2019_2.png)

数据集:    WebQuestionSP       ComplexWebQuestions SimpleQuestions
数据集对应Knowledge Base:Freebase，Freebase，FB2M (subset)
使用了问题中除了实体与关系外的部分用来与知识图谱匹配，会从训练集的gt path中计算rel word与q word的互信息，进而增加q中信息(river mouth就是后选入的)，在链接到知识图谱上的实体/关系后，对这些topic units进行排序，再进行子图上的relation path排序，得到结果。eval结果来看在CWQ，SP以及WQSP的hit1上sota。

> 《Bidirectional Attentive Memory Networks for Question Answering over Knowledge Bases  link》
![image](https://github.com/lemonadeseason/KBQA-Survey/blob/master/NAACL2019_BAMnet.PNG)
本文改进了信息抽取来做KBQA的方法。现有的基于信息抽取的方法大多将问题和KB子图分别emcode，作者认为先验知识（即KB里的知识）可以帮助更好的理解question，同样question本身也可以使得我们关注到KB子图里重要的部分。基于以上想法，作者提出了Bidirectional Attentive Memory network（BAMnet），可以捕捉到问题和KB中重要的信息。在BAMnet网络之上，作者另外使用了two-way attention，帮助模型进一步得到更好的question和KB的representation。最终模型在webquestions上取得了比现有基于信息抽取更好的指标。 

> 《Modeling Semantics with Gated Graph Neural Networks for Knowledge Base Question Answering》
之前基于语义解析的方法大部分都在“如何选择问题对应最合适的semantic parse”，却忽略了在semantic parse本身结构的研究。复杂问题的semantic parse由多个entity和relation组成，比起简单问题的表示要难很多，因此本文着手解决复杂问题的semantic parse结构。作者提出使用Gated Graph Neural Networks以图的结构来编码语法解析结构。

## 4. 相关链接

[KGQA资源总结](https://github.com/BshoterJ/awesome-kgqa)

[知识图库资源汇总 知识库问答-KBQA模块](https://github.com/husthuke/awesome-knowledge-graph)

[KBQA系统总结](https://naotu.baidu.com/file/5c17a01de73d972501d8b3cd187908cb?token=b9a47b442d527efe)

[KBQA知识库问答领域研究综述](https://blog.csdn.net/u012892939/article/details/79451978)

[openKG中文开放知识图谱](http://openkg.cn/home)

[CCKS 2019中文知识图谱问答](https://www.biendata.com/competition/ccks_2019_6/)

[NLPCC2019](http://tcci.ccf.org.cn/conference/2019/cfpt.php)

## 5. 参考资源
[综述文章《introduction to neural network based approaches for question answering over knowledge graphs》](https://arxiv.org/pdf/1907.09361.pdf)

[揭开知识库问答KB-QA的神秘面纱](https://zhuanlan.zhihu.com/p/27141786)
