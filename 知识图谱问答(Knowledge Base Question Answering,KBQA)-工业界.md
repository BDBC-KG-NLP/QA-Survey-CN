# KBQA调研--工业界

## 目录

&emsp;&emsp;[1.&nbsp;任务](#1-任务)

&emsp;&emsp;&emsp;&emsp;[1.1&nbsp;知识图谱背景](#11-知识图谱背景)

&emsp;&emsp;&emsp;&emsp;[1.2&nbsp;任务定义](#12-任务定义)

&emsp;&emsp;&emsp;&emsp;[1.3&nbsp;数据集](#13-数据集)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[1.3.1&nbsp;工业界知识图谱](#131-工业界知识图谱)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[1.3.2&nbsp;CCKS开放域知识图谱问答比赛数据集](#132-CCKS开放域知识图谱问答比赛数据集)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[1.3.3&nbsp;NLPCC开放域知识图谱问答比赛数据集](#133-NLPCC开放域知识图谱问答比赛数据集)

&emsp;&emsp;&emsp;&emsp;[1.4&nbsp;SOTA&nbsp;Board](#14-SOTA-Board)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[1.4.1&nbsp;CCKS2020](#141-CCKS2020)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[1.4.2&nbsp;CCKS2019](#142-CCKS2019)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[1.4.3&nbsp;CCKS2018](#143-CCKS2018)

&emsp;&emsp;&emsp;&emsp;[1.5&nbsp;评测标准](#15-评测标准)

&emsp;&emsp;[2.&nbsp;方法总结](#2-方法总结)

&emsp;&emsp;&emsp;&emsp;[2.1&nbsp;基于规则的方法（github上demo项目的多数方法）](#21-基于规则的方法（github上demo项目的多数方法）)

&emsp;&emsp;&emsp;&emsp;[2.2&nbsp;基于信息抽取的方法（CCKS和NLPCC比赛中方法总结）](#22-基于信息抽取的方法（CCKS和NLPCC比赛中方法总结）)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[2.2.1&nbsp;基于实体与关系识别的模型](#221-基于实体与关系识别的模型)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[2.2.2&nbsp;基于路径匹配的模型](#222-基于路径匹配的模型)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[2.2.3&nbsp;集成模型](#223-集成模型)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[2.2.4&nbsp;每个模型中子任务的实现方式总结](#224-每个模型中子任务的实现方式总结)

&emsp;&emsp;[3.&nbsp;Paper&nbsp;List](#3-Paper-List)

&emsp;&emsp;&emsp;&emsp;[3.1&nbsp;NLPCC+CCKS论文列表](#31-NLPCCCCKS论文列表)

&emsp;&emsp;&emsp;&emsp;[3.2&nbsp;NLPCC+CCKS论文解读](#32-NLPCCCCKS论文解读)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[3.2.1&nbsp;CCKS论文集&笔记](#321-CCKS论文集笔记)

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[3.2.2&nbsp;NLPCC论文集&笔记](#322-NLPCC论文集笔记)

&emsp;&emsp;[4.&nbsp;相关学习资料](#4-相关学习资料)

&emsp;&emsp;&emsp;&emsp;[4.1&nbsp;知识图谱Talk&nbsp;Slides](#41-知识图谱Talk-Slides)

&emsp;&emsp;&emsp;&emsp;[4.2&nbsp;部分企业知识图谱说明](#42-部分企业知识图谱说明)

&emsp;&emsp;[5.&nbsp;参考资源](#5-参考资源)

## 1. 任务

### 1.1 知识图谱背景

知识图谱(Knowledge Base / Knowledge Graph)中包括三类元素：实体（entity）、关系（relation），以及属性（literal）。实体代表一些人或事物，关系用于连接两个实体，表征它们之间的一些联系，如实体Michael Crichton与实体Chicago之间就可以由关系bornin连接，代表作家Michael Crichton出生于城市Chicago。同时，关系不仅可以用于连接两个实体，也可以连接实体和某属性，如关系area可用于连接Chicago和属性606km2，表明chicago面积为606km2。 用更形式化的语言来描述：KB可以表示为三元组的集合，三元组为（entity，relation，entity/literal）。

### 1.2 任务定义

工业界的KBQA系统目的是为用户提供一个用自然语言来提问的界面，使用他们的自己的术语以及语言习惯，通过查询知识图谱得到一个简洁精确的答案。[Introduction to Neural Network based Approaches for Question Answering over Knowledge Graphs](https://arxiv.org/pdf/1907.09361.pdf)



![img](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/pictures/KBQA-example.JPEG)



### 1.3 数据集

没有固定的工业界数据集，工业界的图谱都是通过自己领域内累积的业务数据或是自己建立的app数据进行整理，同时基于产品需求进行建立的，针对图谱的问题多来自企业客服或是app收集的问题，这部分来源未知。以下对工业界使用的知识图谱做一个总括说明，同时介绍两个权威的中文知识图谱问答比赛CCKS和NLPCC所用的知识库和数据集。

#### 1.3.1 工业界知识图谱

工业界的知识图谱有两种分类方式，第一种是根据**领域的覆盖范围不同**分为通用知识图谱和领域知识图谱。其中通用知识图谱注重知识广度，领域知识图谱注重知识深度。通用知识图谱常常覆盖生活中的各个领域，从衣食住行到专业知识都会涉及，但是在每个领域内部的知识体系构建不是很完善；而领域知识图谱则是专注于某个领域(金融、司法等)，结合领域需求与规范构建合适的知识结构以便进行领域内精细化的知识存储和问答。代表的知识图谱分别有：

- 通用知识图谱
  - Google Knowledge Graph
  - Microsoft Satori & Probase
- 领域知识图谱
  - Facebook 社交知识图谱
  - Amazon 商品知识图谱
  - 阿里巴巴商品知识图谱
  - [上海交大学术知识图谱](https://www.acemap.info/)

第二种分类方式是按照**回答问题需要的知识类别**来定义的，分为常识知识图谱和百科全书知识图谱。针对常识性知识图谱，我们只会挖掘问题中的词之间的语义关系，一般而言比较关注的关系包括 isA Relation、isPropertyOf Relation，问题的答案可能根据情景不同而有不同，所以回答正确与否往往存在概率问题。而针对百科全书知识图谱，我们往往会定义很多谓词，例如DayOfbirth, LocatedIn, SpouseOf 等等。这些问题即使有多个答案，这些答案往往也都是确定的，所以构建这种图谱在做问答时最优先考虑的就是准确率。代表的知识图谱分别有：

- 常识知识图谱
  - WordNet, KnowItAll, NELL, Microsoft Concept Graph
- 百科全书知识图谱
  - Freebase, Yago, Google Knowledge Graph

#### 1.3.2 CCKS开放域知识图谱问答比赛数据集

- 问题类型：简单问题：复杂问题（多跳推理问题）=1：1
- 训练集：2298
- 验证集：766
- 测试集：766
- 资源地址：[知识库 密码(huc8)](https://pan.baidu.com/share/init?surl=MOv9PCTcALVIiodUP4bQ2Q)，[问答集](https://github.com/duterscmy/ccks2019-ckbqa-4th-codes/tree/master/data)

#### 1.3.3 NLPCC开放域知识图谱问答比赛数据集

- 问题类型：简单问题（单跳问题）
- 训练集：14609
- 验证集 + 测试集：9870
- 资源地址：[知识库](https://pan.baidu.com/s/1dEYcQXz)，[问答集](http://tcci.ccf.org.cn/conference/2018/taskdata.php)

### 1.4 SOTA Board

#### 1.4.1 CCKS2020 新冠百科知识图谱问答评测

| 名次 | 队伍                   | F1      | 论文链接                                                     | 参考解读/源代码                               |
| ---- | ---------------------- | ------- | ------------------------------------------------------------ | --------------------------------------------- |
| 1    | Artemis（KingSoft AI） | 0.86078 | [基于特征融合的中文知识库问答方法](https://bj.bcebos.com/v1/conference/ccks2020/eval_paper/ccks2020_eval_paper_1_4_1.pdf) |                                               |
| 2    | see（美团）            | 0.85474 | [基于预训练语言模型的检索-匹配式知识图谱问答系统](https://bj.bcebos.com/v1/conference/ccks2020/eval_paper/ccks2020_eval_paper_1_4_2.pdf) |                                               |
| 3    | MiQa（小米）           | 0.85453 | [An Integrated Path Formulation Method for Open Domain Question Answering over Knowledge Base](https://bj.bcebos.com/v1/conference/ccks2020/eval_paper/ccks2020_eval_paper_1_4_3.pdf) |                                               |
| 11   | 难民之路               | 0.54579 | [One Model Structure for All Sub-Tasks KBQA System](https://bj.bcebos.com/v1/conference/ccks2020/eval_paper/ccks2020_eval_paper_1_4_11.pdf) | https://github.com/BettyHcZhang/KBQA_AllenNLP |

#### 1.4.2 CCKS2019

| 名次 | 队伍                           | F1      | 论文链接                                                     | 参考解读/源代码                                       |
| ---- | ------------------------------ | ------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| 1    | jchl (百度智珠尹存祥团队)      | 0.73545 | [混合语义相似度的中文知识图谱问答系统](https://conference.bj.bcebos.com/ccks2019/eval/webpage/pdfs/eval_paper_6_1.pdf) |                                                       |
| 2    | hlt217 (SUDA-HUAWEI)           | 0.73075 | [Combining Neural Network Models with Rules for Chinese Knowledge Base Question Answering](https://conference.bj.bcebos.com/ccks2019/eval/webpage/pdfs/eval_paper_6_2.pdf) |                                                       |
| 3    | 网易互娱AIlab-陈垚鑫           | 0.72514 |                                                              |                                                       |
| 4    | baseline (平安人寿AI-FudanSDS) | 0.70448 | [Multi-Module System for Open Domain Chinese Question Answering over Knowledge Base](https://conference.bj.bcebos.com/ccks2019/eval/webpage/pdfs/eval_paper_6_3.pdf) | https://zhuanlan.zhihu.com/p/92317079                 |
| 5    | DUTIR                          | 0.67683 | [DUTIR中文开放域知识库问答评测报告](https://conference.bj.bcebos.com/ccks2019/eval/webpage/pdfs/eval_paper_6_4.pdf) | https://github.com/duterscmy/ccks2019-ckbqa-4th-codes |

#### 1.4.3 CCKS2018

| 名次 | 队伍          | F1      | 论文链接                                                     | 参考解读/源代码                         |
| ---- | ------------- | ------- | ------------------------------------------------------------ | --------------------------------------- |
| 1    | Keenpower     | 0.66609 | [A QA search algorithm based on the fusion integration of text similarity and graph computation⋆](http://ceur-ws.org/Vol-2242/paper13.pdf) | https://github.com/songlei1994/ccks2018 |
| 2    | KPQA          | 0.66359 |                                                              |                                         |
| 3    | zhanghualiang | 0.57666 |                                                              |                                         |
| 4    | LEQA          | 0.57666 | [A Joint Model of Entity Linking and Predicate Recognition for Knowledge Base Question Answering](http://ceur-ws.org/Vol-2242/paper14.pdf) |                                         |
| 5    | xiehe         | 0.57266 |                                                              |                                         |
| 6    | huawencai     | 0.57051 |                                                              |                                         |
| 7    | COINQA        | 0.56930 | [Semantic Parsing for Multiple-relation Chinese Question Answering](http://ceur-ws.org/Vol-2242/paper15.pdf) |



### 1.5 评测标准

这个企业内部没有具体写明，提到的都在学术界有使用，这里给出CCKS和NLPCC比赛的测评方法

- **Mean Reciprocal Rank (MRR)**

  <a href="https://www.codecogs.com/eqnedit.php?latex=MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}" title="MRR=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{rank_i}" /></a>

  - |Q|代表问题总数，rank_i代表第一个正确的答案在答案集合C_i中的位置
  - 如果C_i中没有正确答案，<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{rank_i}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{rank_i}=0" title="\frac{1}{rank_i}=0" /></a>

- **Accuracy@N**

  <a href="https://www.codecogs.com/eqnedit.php?latex=Accuracy@N=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\delta(C_i,A_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Accuracy@N=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\delta(C_i,A_i)" title="Accuracy@N=\frac{1}{|Q|}\sum_{i=1}^{|Q|}\delta(C_i,A_i)" /></a>

  - 当答案集合C_i中至少有一个出现在gold answerA_i中，<a href="https://www.codecogs.com/eqnedit.php?latex=\delta(C_i,A_i)=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta(C_i,A_i)=1" title="\delta(C_i,A_i)=1" /></a>，否则为0

- **Averaged F1**

  <a href="https://www.codecogs.com/eqnedit.php?latex=AveragedF1=\frac{1}{|Q|}\sum_{i=1}^{|Q|}F_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?AveragedF1=\frac{1}{|Q|}\sum_{i=1}^{|Q|}F_i" title="AveragedF1=\frac{1}{|Q|}\sum_{i=1}^{|Q|}F_i" /></a>

  - F_i是Q_i问题产生答案的F1值，如果A_i和C_i无交集F1为0



## 2. 方法总结

### 2.1 基于规则的方法（github上demo项目的多数方法）

代表项目：[豆瓣影评问答](https://github.com/weizhixiaoyi/DouBan-KGQA) 

- 实体识别
  - 词表 ，字符串相似度（或BiLSTM-CRF）
- 属性链接
  - 词表，字符串相似度 （或CNN等分类模型）
- 答案推理
  - 规则模板转换得到SPARQL语句
  - （或TransE表示学习的方法进行答案推理）

### 2.2 基于信息抽取的方法（CCKS和NLPCC比赛中方法总结）

#### 2.2.1 基于实体与关系识别的模型

这种模型首先通过**实体识别**等方法获得问句中实体的 mention，再通过主办方提供的数据库中 mention2ent 及其他信息通过**实体链接** mention 到数据库中对应实体列表，通过多种设计的打分函数最终选取一个或多个与问句最相关的实体作为核心实体，用于之后生成关系、查询语句等。

在获得中心实体后，此类系统先得到该实体在数据库中的邻居子图，并抽取出邻居子图中所有的关
系作为候选。对于这些关系，系统同样使用设计好的打分函数来评估每一者与问题的契合性（**关系模型**）

同时，此类模型也大多采用一个**问题类型识别**模块以提高表现。数据集中的问题可以分为单跳问
题与多跳问题，多跳问题在查询图结构上也有多种形式；通过训练，问题类型识别模型能够将问句分类到相应的查询图结构模板上。这些有了这些模板信息，便可以根据之前获得的关系取打分高的一个（如果问题类型为单跳）或多个（如果类型为多跳）作为结果，将其与中心实体对应的查询路径填入最终查询数据库得到答案。

#### 2.2.2 基于路径匹配的模型

与上一个模型类似，同样需要**实体识别**识别出问题的中心实体，通过问**题类别识别**确定问题的类型，同时通过**实体链接**将中心实体链接到知识图谱中的实体中。而在**关系模型**中，本方法并不直接选取最优关系，而是直接获取所有能匹配上问题类型模板的，在中心实体周边的查询路径；之后，通过各类基于问句与查询路径特征来给后者打分，并选取最优查询路径的模型，最终答案便可以求得。

#### 2.2.3 集成模型

按照上述的**规则方法**，**实体与关系识别方法**以及**路径匹配方法**同时进行回答，在答案选取的时候选择置信度高的进行回答。

#### 2.2.4 每个模型中子任务的实现方式总结

- **实体识别**

  这部分主要由自定义的识别规则和神经网络训练的方法构成

  - 规则
    - 自定义字典(分词，词频，倒排索引)识别
    - 建立停用词表删去无用词
    - 分词后的词与知识图谱的实体的字符串匹配(jaccord，编辑距离)
  - 辅助工具
    - NER工具包识别问句中人名，地点，机构等
  - 神经网络
    - Bert/BiLSTM + CRF

- **实体链接**

  主要方法是通过选用一部分特征作为特征参数（a1, a2, ..., an），使用线性打分排序(s=k1a1 + k2a1 + ... + knan)或者简单机器学习算法进行排序。

  - 选用特征
    - 实体名称和问题的字符串匹配度(char/word)
    - 图谱子图与问题的匹配度
    - 实体类型与问题的匹配度
    - 实体长度
    - 实体在图谱中关系个数/出现频率
    - 实体与疑问词距离
    - ...
  - 模型
    - lambdarank
    - xgboost
    - logistic regression

- **关系模型1：关系识别与排序**(1/2hop)

  - 关系和问题的语义相似度（bert-bilstm-fc-cosine）
  - 关系值和问题的语义相似度（bert-bilstm-fc-cosine）
  - 关系和问题的字符覆盖率

- **关系模型2：路径排序**

  - 将链接到的实体和 **实体的1/2跳关系** 组成路径，通过bert-similarity模型进行训练
  - 路径与问题的jaccard，编辑距离
  - 自身定义的模板匹配度...

- **关系模型3：规则匹配排序**

  - 将问题分割为多个部分，参照word/phrases in the kb + existing word segmentation tools，将问题分为各部分都和kb中实体/属性/关系相似的部分，按分值高低与知识图谱对应部分进行链接。
  - 将问题分为单跳，多跳类型（共8种），记录下各自的结构，与问题中分割出的问题结构进行相似度比较

- **问题类型识别**

  - bert/cnn/rnn分类模型确定单跳/多跳种类

## 3. Paper List

### 3.1 NLPCC+CCKS论文列表

[1] [NLPCC2015 1st](http://tcci.ccf.org.cn/conference/2015/papers/246.pdf) Ye Z, Jia Z, Yang Y, et al. Research on open domain question answering system[M]//Natural Language Processing and Chinese Computing. Springer, Cham, 2015: 527-540.

[2] [NLPCC2016 1st](https://link.springer.com/chapter/10.1007%2F978-3-319-50496-4_65) Lai Y, Lin Y, Chen J, et al. Open domain question answering system based on knowledge base[M]//Natural Language Understanding and Intelligent Applications. Springer, Cham, 2016: 722-733.

[3] [NLPCC2016 2nd](https://link.springer.com/chapter/10.1007/978-3-319-50496-4_86) Yang F, Gan L, Li A, et al. Combining deep learning with information retrieval for question answering[M]//Natural Language Understanding and Intelligent Applications. Springer, Cham, 2016: 917-925.

[4] [NLPCC2016 3rd](https://link.springer.com/chapter/10.1007/978-3-319-50496-4_25) Xie Z, Zeng Z, Zhou G, et al. Knowledge base question answering based on deep learning models[M]//Natural Language Understanding and Intelligent Applications. Springer, Cham, 2016: 300-311.

[5] [NLPCC2016 4th](https://link.springer.com/chapter/10.1007/978-3-319-50496-4_82) Wang L, Zhang Y, Liu T. A deep learning approach for question answering over knowledge base[M]//Natural Language Understanding and Intelligent Applications. Springer, Cham, 2016: 885-892.

[6] [NLPCC2017 1st](http://tcci.ccf.org.cn/conference/2017/papers/2003.pdf) Lai Y, Jia Y, Lin Y, et al. A Chinese question answering system for single-relation factoid questions[C]//National CCF Conference on Natural Language Processing and Chinese Computing. Springer, Cham, 2017: 124-135.

[7] [NLPCC2017 2nd](http://tcci.ccf.org.cn/conference/2017/papers/2041.pdf) Zhang H, Zhu M, Wang H. A Retrieval-Based Matching Approach to Open Domain Knowledge-Based Question Answering[C]//National CCF Conference on Natural Language Processing and Chinese Computing. Springer, Cham, 2017: 701-711.

[8] [NLPCC2017 会议](http://xbna.pku.edu.cn/CN/10.13209/j.0479-8023.2017.155) 周博通, 孙承杰, 林磊, et al. 基于LSTM的大规模知识库自动问答[J]. 北京大学学报：自然科学版, 2018.

[9] [NLPCC2018 1st](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_35) Ni H, Lin L, Xu G. A Relateness-Based Ranking Method for Knowledge-Based Question Answering[C]//CCF International Conference on Natural Language Processing and Chinese Computing. Springer, Cham, 2018: 393-400.

[10] [CCKS2018 1st](http://ceur-ws.org/Vol-2242/paper13.pdf) A QA Search Algorithm based on the Fusion Integration of Text Similarity and Graph Computation

[11] [CCKS2018 2nd](http://ceur-ws.org/Vol-2242/paper14.pdf) A Joint Model of Entity Linking and Predicate Recognition for Knowledge Base Question Answering

[12] [CCKS2018 3rd](http://ceur-ws.org/Vol-2242/paper15.pdf) Semantic Parsing for Multiple-relation Chinese Question Answering

[13] [CCKS2019 1st](https://conference.bj.bcebos.com/ccks2019/eval/webpage/pdfs/eval_paper_6_1.pdf) 混合语义相似度的中文知识图谱问答系统

[14] [CCKS2019 2nd](https://conference.bj.bcebos.com/ccks2019/eval/webpage/pdfs/eval_paper_6_2.pdf) Combining Neural Network Models with Rules for Chinese Knowledge Base Question Answering

[15] [CCKS2019 3rd](https://conference.bj.bcebos.com/ccks2019/eval/webpage/pdfs/eval_paper_6_3.pdf) Multi-Module System for Open Domain Chinese Question Answering over Knowledge Base

[16] [CCKS2019 4th](https://conference.bj.bcebos.com/ccks2019/eval/webpage/pdfs/eval_paper_6_4.pdf) DUTIR中文开放域知识库问答评测报告

### 3.2 NLPCC+CCKS论文解读

#### 3.2.1 CCKS论文集&笔记

- [2019](https://github.com/BDBC-KG-NLP/KBQA-Survey/tree/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019)
- [2018](https://github.com/BDBC-KG-NLP/KBQA-Survey/tree/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2018)

#### 3.2.2 NLPCC论文集&笔记

- [2018](https://github.com/BDBC-KG-NLP/KBQA-Survey/tree/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/NLPCC/NLPCC2018)
- [2017](https://github.com/BDBC-KG-NLP/KBQA-Survey/tree/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/NLPCC/NLPCC2017)

## 4. 相关学习资料

### 4.1 知识图谱Talk Slides

- [CCKS slides](https://github.com/BDBC-KG-NLP/KBQA-Survey/tree/master/KBQA%20Industry/CCKS%2BNLPCC%20slides/CCKS)
- [NLPCC slides](https://github.com/BDBC-KG-NLP/KBQA-Survey/tree/master/KBQA%20Industry/CCKS%2BNLPCC%20slides/NLPCC)

### 4.2 部分企业知识图谱说明

- [Google](https://www.blog.google/products/search/introducing-knowledge-graph-things-not/)
- [Bing](https://blogs.bing.com/search-quality-insights/2017-07/bring-rich-knowledge-of-people-places-things-and-local-businesses-to-your-apps)
- [Uber](https://eng.uber.com/uber-eats-query-understanding/)
- [Amazon](https://blog.aboutamazon.com/innovation/making-search-easier)
- [Airbnb](https://medium.com/airbnb-engineering/scaling-knowledge-access-and-retrieval-at-airbnb-665b6ba21e95)
- [eBay](https://www.ebayinc.com/stories/news/cracking-the-code-on-conversational-commerce/)
- [Linkedin](https://engineering.linkedin.com/blog/2016/10/building-the-linkedin-knowledge-graph)
- [Accenture](https://www.accenture.com/us-en/insights/digital/data-to-knowledge)
- [Bloomberg](https://speakerdeck.com/emeij/understanding-news-using-the-bloomberg-knowledge-graph)
- [Maana](https://engineering.linkedin.com/blog/2016/10/building-the-linkedin-knowledge-graph)
- [阿里巴巴1](http://www.elecfans.com/d/908682.html)，[阿里巴巴2](https://www.sohu.com/a/168239286_629652)
- [腾讯云知文](https://max.book118.com/html/2019/0122/8023043054002003.shtm)
- [百度](https://baijiahao.baidu.com/s?id=1643915882369765998&wfr=spider&for=pc)
- [美团KG构建方法](https://tech.meituan.com/2018/11/01/meituan-ai-nlp.html)，[美团餐饮娱乐知识图谱](https://tech.meituan.com/2018/11/22/meituan-brain-nlp-01.html)，[美团BERT](https://tech.meituan.com/2019/11/14/nlp-bert-practice.html)

## 5. 参考资源

[中文知识图谱问答 CCKS2019 CKBQA 参赛总结](https://blog.csdn.net/zzkv587/article/details/102954876)

[CCKS2019 测评报告](https://arxiv.org/ftp/arxiv/papers/2003/2003.03875.pdf)
