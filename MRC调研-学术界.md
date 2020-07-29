# MRC调研——学术界

## 1.任务

### 1.1 背景

 机器阅读理解，又称阅读理解问答，要求机器阅读并理解人类自然语言文本，在此基础上，解答跟文本信息相关的问题。该任务通常被用来衡量机器自然语言理解能力，可以帮助人类从大量文本中快速聚焦相关信息，降低人工信息获取成本，在文本问答、信息抽取、对话系统等领域具有极强的应用价值。近年来，机器阅读理解受到工业界和学术界越来越广泛的关注，是自然语言处理领域的研究热点之一。 

### 1.2 任务定义

 机器阅读理解（Machine Reading Comprehension, MRC），通过给定上下文，要求机器根据上下文回答问题，来测试机器理解自然语言的程度。常见的任务分为：完形填空（Cloze Test）、多项选择（Multiple Choice）、片段抽取（Span Extraction）、自由回答（ Free Answering）。近年来，学界考虑到当前方法的局限性，MRC出现了新的任务，比如基于知识的机器阅读理解（Knowledge-Based MRC），不可答问题的机器阅读理解（MRC with Unanswerable Questions），多文章机器阅读理解（Multi-passage MRC），口语问题回答（Conversational Question Answering）。 

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

- 由 Microsoft于 2013年在《 MCTest: A Challenge Dataset for the Open-Domain Machine Comprehension of Text》一文中发布
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

|      数据集      |              Paper               |                         Data                         |
| :--------------: | :------------------------------: | :--------------------------------------------------: |
| CNN & Daily Mail | https://arxiv.org/abs/1506.03340 |            https://cs.nyu.edu/~kcho/DMQA/            |
|       CBT        | https://arxiv.org/abs/1511.02301 |       https://research.fb.com/downloads/babi/        |
|     LAMBADA      |                                  |                                                      |
|   Who-did-What   |                                  | https://tticnlp.github.io/who_did_what/download.html |
|      CLOTH       |                                  |                                                      |
|      CliCR       |                                  |                                                      |

 **Multiple Choice** 任务数据集

| 数据集 |                            Paper                             |                             Data                             |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| MCTest | https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/ | https://github.com/mcobzarenco/mctest/tree/master/data/MCTest |
|  RACE  |               https://arxiv.org/abs/1704.04683               |           https://www.cs.cmu.edu/~glai1/data/race/           |

 **Span Extraction** 任务数据集

| 数据集   | Paper                            | Data                                        |
| -------- | -------------------------------- | ------------------------------------------- |
| SQuAD    | https://arxiv.org/abs/1606.05250 | https://rajpurkar.github.io/SQuAD-explorer/ |
| NewsQA   | https://arxiv.org/abs/1611.09830 | https://github.com/Maluuba/newsqa           |
| TriviaQA |                                  |                                             |
| DuoRC    |                                  |                                             |

 **Free Answering** 任务数据集

| 数据集      | Paper                            | Data                                    |
| ----------- | -------------------------------- | --------------------------------------- |
| bAbI        | https://arxiv.org/abs/1611.09268 | https://research.fb.com/downloads/babi/ |
| MS MARCO    | https://arxiv.org/abs/1611.09268 | https://microsoft.github.io/msmarco/    |
| SearchQA    |                                  |                                         |
| NarrativeQA |                                  |                                         |
| DuReader    |                                  |                                         |

### 1.4 SOTA

**SQuAD**

SA-Net on Albert (ensemble)	90.724	EM

SA-Net on Albert (ensemble)	93.011 F1

https://rajpurkar.github.io/SQuAD-explorer/

**MS MARCO Question Answering**

Multi-doc Enriched BERT Ming Yan of Alibaba Damo NLP   0.540 Rouge-L

Multi-doc Enriched BERT Ming Yan of Alibaba Damo NLP   0.565 Bleu-1

https://microsoft.github.io/msmarco/

### 1.5 评测标准

- Accuracy
- average f1
- rouge-L
- bleu

## 2 方法总结

大致可以划分为两类：基于规则的方法，基于深度学习的方法

### 2.1 基于规则的方法

​         通过人工制定规则，对文章中的候选语句和问题进行匹配度打分，然后选取得分最高的候选语句作为答案。

### 2.2 基于深度学习的方法

- 词向量模块（Embeddings）：传统的词表示（One-hot，Distributed）；预训练上下文表示；添加更多信息（字向量、POS、NER、词频、问题类别）
- 编码模块（Feature Extraction）：RNN（LSTM，GRN）；CNN
- 注意力模块：单路注意力模型、双路注意力模型、自匹配注意力模型
- 答案预测模块（Answer Prediction）：word predictor，option selector，span extractor，answer generator

## 3 Paper List

### 3.1 经典论文

| 会议名称  | 论文名称                                                     | 下载链接 |
| --------- | ------------------------------------------------------------ | -------- |
| NIPS 2015 | Teaching machines to read and comprehend                     |          |
| ACL 2016  | Text understanding with the attention sum reader network     |          |
| ACL 2016  | A Through Examination of the CNN_Daily Mail Reading Comprehension Task |          |
| ACL 2017  | Attention-over-Attention Neural Networks for Reading Comprehension |          |
| ICLR 2017 | Bidirectional Attention Flow for Machine Comprehension       |          |
| ACL 2017  | Gated Self-Matching Networks for Reading Comprehension and Question Answering |          |
| ACL 2018  | Simple and Effective Multi-Paragraph Reading Comprehension   |          |

### 3.2 近3年论文

| 会议名称   | 论文名称                                                     | 下载链接 |
| ---------- | ------------------------------------------------------------ | -------- |
| AAAI 2020  | SG-Net: Syntax-Guided Machine Reading Comprehension.         |          |
| AAAI 2020  | Generating Well-Formed Answers by Machine Reading with Stochastic Selector Networks. |          |
| AAAI 2020  | A Robust Adversarial Training Approach to Machine Reading Comprehension. |          |
| AAAI 2020  | Multi-Task Learning with Generative Adversarial Training for Multi-Passage Machine Reading Comprehension. |          |
| AAAI 2020  | Distill BERT to Traditional Models in Chinese Machine Reading Comprehension (Student Abstract). |          |
| AAAI 2020  | Assessing the Benchmarking Capacity of Machine Reading Comprehension Datasets. |          |
| AAAI 2020  | A Multi-Task Learning Machine Reading Comprehension Model for Noisy Document (Student Abstract) |          |
| AAAI 2020  | Rception: Wide and Deep Interaction Networks for Machine Reading Comprehension (Student Abstract). |          |
| AAAI 2019  | Read + Verify: Machine Reading Comprehension with Unanswerable Questions. |          |
| AAAI 2019  | Teaching Machines to Extract Main Content for Machine Reading Comprehension. |          |
| AAAI 2018  | Byte-Level Machine Reading Across Morphologically Varied Languages. |          |
| AAAI 2018  | S-Net: From Answer Extraction to Answer Synthesis for Machine Reading Comprehension. |          |
| EMNLP 2019 | Incorporating External Knowledge into Machine Reading for Generative Question Answering. |          |
| EMNLP 2019 | Cross-Lingual Machine Reading Comprehension.                 |          |
| EMNLP 2019 | A Span-Extraction Dataset for Chinese Machine Reading Comprehension. |          |
| EMNLP 2019 | Cosmos QA: Machine Reading Comprehension with Contextual Commonsense Reasoning. |          |
| EMNLP 2019 | Towards Machine Reading for Interventions from Humanitarian-Assistance Program Literature. |          |
| EMNLP 2019 | Revealing the Importance of Semantic Retrieval for Machine Reading at Scale. |          |
| EMNLP 2019 | Machine Reading Comprehension Using Structural Knowledge Graph-aware Network. |          |
| EMNLP 2019 | NumNet: Machine Reading Comprehension with Numerical Reasoning. |          |
| EMNLP 2019 | Adversarial Domain Adaptation for Machine Reading Comprehension. |          |
| ACL 2020   | Explicit Memory Tracker with Coarse-to-Fine Reasoning for Conversational Machine Reading. |          |
| ACL 2020   | Recurrent Chunking Mechanisms for Long-Text Machine Reading Comprehension. |          |
| ACL 2020   | A Frame-based Sentence Representation for Machine Reading Comprehension. |          |
| ACL 2020   | Machine Reading of Historical Events.                        |          |
| ACL 2020   | A Self-Training Method for Machine Reading Comprehension with Soft Evidence Extraction. |          |
| ACL 2020   | Enhancing Answer Boundary Detection for Multilingual Machine Reading Comprehension. |          |
| ACL 2020   | Document Modeling with Graph Attention Networks for Multi-grained Machine Reading Comprehension. |          |
| ACL 2019   | Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading. |          |
| ACL 2019   | Explicit Utilization of General Knowledge in Machine Reading Comprehension. |          |
| ACL 2019   | Enhancing Pre-Trained Language Representations with Rich Knowledge for Machine Reading Comprehension. |          |
| ACL 2019   | MC\2: Multi-perspective Convolutional Cube for Conversational Machine Reading Comprehension. |          |
| ACL 2019   | E3: Entailment-driven Extracting and Editing for Conversational Machine Reading. |          |
| ACL 2019   | Learning to Ask Unanswerable Questions for Machine Reading Comprehension. |          |
| ACL 2018   | Stochastic Answer Networks for Machine Reading Comprehension. |          |
| ACL 2018   | Jack the Reader - A Machine Reading Framework.               |          |
| ACL 2018   | Multi-Passage Machine Reading Comprehension with Cross-Passage Answer Verification. |          |
| ACL 2018   | Multi-Relational Question Answering from Narratives: Machine Reading and Reasoning in Simulated Worlds. |          |
| IJCAI 2020 | LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning. |          |
| IJCAI 2020 | An Iterative Multi-Source Mutual Knowledge Transfer Framework for Machine Reading Comprehension. |          |
| IJCAI 2020 | Asking Effective and Diverse Questions: A Machine Reading Comprehension based Framework for Joint Entity-Relation Extraction. |          |
| IJCAI 2018 | Reinforced Mnemonic Reader for Machine Reading Comprehension. |          |
| CIKM 2019  | Machine Reading Comprehension: Matching and Orders.          |          |
| CIKM 2018  | An Option Gate Module for Sentence Inference on Machine Reading Comprehension. |          |

## 4 相关链接

## 5 参考资源

自然语言处理常用数据集： https://zhuanlan.zhihu.com/p/46834868 

《Neural Machine Reading Comprehension Methods and Trends》综述解析： https://zhuanlan.zhihu.com/p/80905984 





