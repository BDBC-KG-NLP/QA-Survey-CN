# CQA调研——学术界

## 目录

  * [1. 任务](#1-任务)
     * [1.1. 背景](#11-背景)
     * [1.2. 任务定义](#12-任务定义)
     * [1.3. 数据集](#13-数据集)
     * [1.4. 评测标准](#14-评测标准)
  * [2. 方法总结](#2-方法总结)
     * [2.1. 基于词频的方法](#21-基于词频的方法)
     * [2.2. 基于语义的方法](#22-基于语义的方法)
        * [2.2.1. 基于表示的方法](#221-基于表示的方法)
        * [2.2.2. 基于交互的方法](#222-基于交互的方法)
  * [3. Paper List](#3-paper-list)
     * [3.1. 论文列表](#31-论文列表)
     * [3.2. 论文解读](#32-论文解读)

## 1. 任务

### 1.1. 背景
智能问答系统已经有70年的发展历史。早期的智能问答系统通常只接受特定形式的自然语言问句，而且可以供智能问答系统进行训练的数据也很少，所以无法进行基于大数据的开放领域的问答从而未被广泛使用。进入九十年代之后，由于互联网的发展，大量可供训练的问答对在网上可以被搜集和找到。尤其是 TREC-QA评测的推出，极大推动促进了智能问答系统的发展。目前，已经有很多智能问答系统产品问世。例如IBM研发的智能问答机器人Watson在美国智力竞赛节目《Jeopardy!》中战胜人了选手，其所拥有的DeepQA 系统集成了统计机器学习、信息抽取、知识库集成和知识推理等深层技术。苹果公司的 Siri 系统和微软公司的cortana 分别在 iPhone 手机中和 Windows10 操作系统中都取得了很好的效果。在国内，众多企业和研究团体也推出了很多以智能问答技术为核心的机器人，例如：微软公司的“小冰”、百度公司的“度秘”和中科汇联公司的“爱客服”，我们可以看到，这些机器人不仅提供情感聊天的闲聊功能，而且还能提供私人秘书和智能客服这样的专业功能。这些智能系统的出现标志着智能问答技术正在走向成熟，预计未来还会有更多功能的机器人问世和解决用户的各种需求。  
随着网络与信息技术的快速发展，网络上产生了海量信息，人们需要更加快速直接获取有用而准确地信息，推动问答系统的发展。根据系统处理的数据格式，问答系统又可以分为：基于结构化数据的问答系统、基于自由文本数据的问答系统、基于问题答案对数据的问答系统。其中基于问题答案对的问答系统主要分为两种，一种是利用各个公司或者组织在网站上提供的常用问题列表(FAQ)FAQ具有限定领域、质量高、组织好等优点，使得系统回答问题的水平大大提高，但FAQ的获取成本高，这个缺点又制约了基于FAQ的问答系统的应用范围。另一种便是利用问答社区中用户自己产生的问答对的社区问答系统（Community Question Answering, CQA）。随着问答社区的兴起，如 Yahoo!Answers， Quora，StackOverflow，百度知道和知乎等，一种新的问答形式开始大量出现。通过使用社区问答系统，人们不但可以发布问题进行提问以满足自己的信息需求，而且还可以回答其他用户提问的问题来分享自己的知识，让用户所拥有的隐性知识转化成显性知识。
### 1.2. 任务定义
对于FAQ问答和CQA问答而言，该任务通常可以可以分为两个子任务：  
**问题-问题匹配**: 给定一个问题和一个候选问题集合,选择与其最相似的问题  
**问题-答案匹配**：给一个问题和一个候选答案集合，选择与其最相似的答案  


### 1.3. 数据集

#### QQ匹配:
-  [**Quora Question Pairs**](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs):是Quora发布的一个数据集，旨在识别重复的问题。它由Quora上超过40万对问题组成，每个问题对都用0/1标签标注是否为重复问题。
- [**MRPC**](https://www.microsoft.com/en-us/download/details.aspx?id=52398&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F607d14d9-20cd-47e3-85bc-a2f65cd28042%2Fdefault.aspx)：MRPC是Microsoft Research Paraphrase Corpus的缩写。它包含从网络新闻源中提取的5,800对句子，以及指示每对是否捕获释义​​/语义对等关系的人工标注。
- [**LCQMC**](http://icrc.hitsz.edu.cn/info/1037/1146.htm):百度发布的一个大型中文问题匹配数据集，数据来自百度知道。每条数据为两个问题和它们的相似性标签（用1/0代表相似/不相似)。

#### QA匹配:
- [**WikiQA**](https://www.microsoft.com/en-us/download/details.aspx?id=52419) :一组公开可用的问题答案对集合，由Microsoft Research收集和注释以用于开放域答案选择问题的研究。
- [**TRECQA**](https://trec.nist.gov/data/qa.html)从TRECQA8-13的数据中搜集整理的数据集，从每个问题的文档库中自动选择候选答案。该数据集是答案句子选择使用最广泛的基准之一。
- [**QNLI**](https://gluebenchmark.com/tasks)：SQuAD数据集的修改版本，允许进行答案选择任务。SQuAD中的上下文段落被分成句子，每个句子都与问题配对。当句子包含答案时，将为问题句子对提供真正的标签。有86,308 / 10,385个问题和428,998 / 169,435个问题/答案对。



### 1.4. 评测标准
- ACC:判断两个句子是否相似的准确率
- P@1:判断排序第一的答案是否正确
- MAP(Mean Average Precision): 评测整个排序的质量。  
![image](https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/MAP1.svg)
其中R表示相关文档的总个数，position(r)表示，结果列表从前往后看，第r个相关文档在列表中的位置。比如，有三个相关文档，位置分别为1、3、6，那么AveP=1/3 * (1/1+2/3+3/6)。
最后，MAP计算所有Query的平均准确率分数：
![image](https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/MAP2.svg)
Q为问题数目总量。
- MRR(Mean Reciprocal Rank)：是把标准答案在被评价系统给出结果中的排序取倒数作为它的准确度，再对所有的问题取平均  
![image](https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/MRR.svg)

### 1.5 Learnboard

| 数据集  | stoa |论文题目|年份|论文链接|code|
| ------------- | ------------- |------------- |------------- |------------- |------------- |
|Quora pairs|90.5(ACC)   |ALBERT: A Lite BERT for Self-supervised Learning of Language Representations   | 2020|https://arxiv.org/pdf/1909.11942v6.pdf |https://github.com/google-research/ALBERT |
|MRPC|91.9(ACC) |StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding|2019|https://arxiv.org/abs/1908.04577 | - |
|LCQMRC|87.9(ACC)|ERNIE 2.0: A Continual Pre-training Framework for Language Understanding|2019|https://arxiv.org/abs/1907.12412v1   |https://github.com/PaddlePaddle/ERNIE |
|Wikiqa|92.0(MAP)|TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection  |2019|https://arxiv.org/pdf/1911.04118.pdf |https://github.com/alexa/wqa_tanda |
|Trecqa|94.3(MAP)|TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection|2019|https://arxiv.org/pdf/1911.04118.pdf |https://github.com/alexa/wqa_tanda |
|QNLI|99.2（ACC)|ALBERT: A Lite BERT for Self-supervised Learning of Language Representations|2020|https://arxiv.org/pdf/1909.11942v6.pdf |https://github.com/google-research/ALBERT |
## 2. 方法总结
可以初步划分为两类，基于词频的方法，基于语义的方法

### 2.1. 基于词频的方法
在机器学习出现之前，传统文本匹配方法通常是根据句子中的词频信息进行检索的，如信息检索中的TF-IDF,BM25，语言模型等方法，主要解决字面相似度问题。这些方法由于计算简单，适用范围广，到现在依旧是很多场景下的优秀基准模型。
#### TF-IDF介绍
TF-IDF（term frequency–inverse document frequency）是一种用于资讯检索与文本挖掘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文档集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。  
TF：在一份给定的文件里，词频（term frequency，TF）指的是某一个给定的词语在该文件中出现的次数。对于在某一特定文件里的词语ti来说，它的重要性可表示为：
TF = 某个词在文档中的出现次数/文档中的总词数  
IDF：逆向文件频率（inverse document frequency，IDF）是一个词语普遍重要性的度量。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到：  
IDF = log(语料库中的总文档数/语料库中出现该词的文档数)  
最终，TF-IDF=TF * IDF  

#### BM25介绍
BM25算法是一种应用广泛的对TF-IDF的改进算法，解决了TF-IDF偏向于长文档的问题。传统的TF值理论上是可以无限大的。而BM25与之不同，它在TF计算方法中增加了一个常量k，用来限制TF值的增长极限。
BM25还引入了平均文档长度的概念，单个文档长度对相关性的影响力与它和平均长度的比值有关系引入另外两个参数：L和b。L是文档长度与平均长度的比值。如果文档长度是平均长度的2倍，则L＝2。b是一个常数，它的作用是规定L对评分的影响有多大。加了L和b的TF计算公式变为:  
TF = ((k + 1) * tf) / (k * (1.0 - b + b * L) + tf)  
IDF部分计算方法与TF-IDF中相同。  
最终，BM25=TF * IDF  

#### 统计语言模型介绍
统计语言模型用于计算给定一个问题，另一个问题由其生成的概率，通过引入马尔可夫假设，我们可以认为一句话中每个单词出现的概率只与它前面n个词有关，即第n个词的出现与前面N-1个词相关，整句的概率就是各个词出现概率的乘积。该模型被称为ngram语言模型。  
统计语言模型通常对语料库的大小有着较强的要求，通常来说，随着n-gram模型中n的增加，模型对于概率的估计会更加准确，但是需要的数据量也会成大大增加，所以，常用的统计语言模型通常为2-gram模型或者one-gram模型。

### 2.2 基于语义的方法
目前，深度学习模型已经在社区问答领域得到了广泛的应用，由于深度模型考虑了问题与问题之间的语义信息，通常比传统的基于词频的模型能取得更好的效果。
#### 2.2.1 基于表示的方法
基于表示的方法已被用于文本匹配任务,包括语义相似性,重复问题检测,自然语言推理。下图显示了基于表示的方法的一般架构。输入句子的矢量表示由编码器分别构建。两个输入句子对彼此表示的计算没有影响。之后，使用余弦相似度，逐元素运算或基于神经网络的组合等方法对编码的向量进行比较。这种体系结构的优势在于，将相同的编码器应用于每个输入语句会使模型更小。另外，句子向量可以用于可视化，句子聚类和许多其他目的。  
将深度学习应用于答案选择的最初尝试之一是提出的词袋模型。该模型通过简单地获取句子中所有单词向量的平均值（先前已删除所有停用词）来生成句子的向量表示。与许多需要大量手工制作的功能或外部资源的传统技术相比，将其他重叠的字数统计功能与模型集成在一起可以提高性能。Severyn和Moschitti等于2015年提出了一个基于CNN的模型，该模型采用卷积神经网络（CNN）生成输入句子的表示形式。 CNN基于先前已应用于许多句子分类任务的架构。该模型中，每个输入句子都使用CNN进行建模，该CNN在多个粒度级别上提取特征并使用多种类型的池化。然后，使用多个相似性指标以几种粒度比较输入语句的表示形式。最后，将比较结果输入到完全连接的层中，以获得最终的相关性得分。
下面介绍来自《LSTM-BASED DEEP LEARNING MODELS FOR NONFACTOID ANSWER SELECTION》文章中一种基于表示的方法QA-LSTM。
QA-LSTM模型采用双向长期短期记忆（biLSTM）网络和池化层来独立构建输入句子的分布式矢量表示。然后，该模型利用余弦相似度来衡量句子表示的距离。 主要可以分为单词表示层，句子表示层，相似度计算层三部分：
1. 单词表示层：
该层的目标是将原始每个词的one-hot编码转换为d维的词向量编码，通常使用word2vec或者glove词向量
2. 句子表示层：
模型采用双向长期短期记忆（biLSTM）网络和池化层来独立构建输入句子的向量表示。之后文章尝试了三种不同的方式来得到最终的句子向量表示：（1）最大池化（2）平均池化（3）两个方向上最后一个词的向量表示的拼接。通过试验，文章最终采用了最大池化的方法得到句子的向量表示
3. 相似度计算层
利用两个句子向量的cosine相似度来得到最终的相似度得分  
训练方法：loss的计算公式如下  
L = max{0, M − cosine(q, a+) + cosine(q, a−)}
其中a+为正确答案，a-为错误答案
![image](https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/基于表示的方法.png)

#### 2.2.2 基于交互的方法
基于比较的方法通常比较输入句子的较小单位（例如单词），然后将比较结果汇总（例如，通过CNN或RNN），以做出最终决定。与基于表示的方法相比，基于比较的方法可以捕获输入句子之间的更多交互功能，因此在对TrecQA等公共数据集进行评估时，通常具有更好的性能。下图显示了来自《Bilateral Multi-Perspective Matching for Natural Language Sentences》一个典型的基于比较的方法的模型。该模型包括以下五层。  
1. 单词表示层（Word Representation Layer）
该层的目标是用d维向量表示输入句子中的每个单词。BiMPM构造具有两个分量的d维向量：一个字符组成的嵌入和一个预先用GloVe或word2vec训练的词嵌入。  
2. 上下文表示层（Contex Representation Layer）
该层的目标是为输入句子中的每个位置获取一个新的表示形式，该表示形式除了捕获该位置的单词以外，还捕获一些上下文信息。 BiMPM使用biLSTM生成上下文表示。  
3. 匹配层（Matching Layer）
该层的目标是将一个句子的每个上下文表示与另一句子的所有上下文表示进行比较。该层的输出是两个匹配向量序列，其中每个匹配向量对应于一个句子的一个位置与另一个句子的所有位置的比较结果。  
4. 聚合层（Aggregation Layer）
该层的目标是汇总来自上一层的比较结果。 BiMPM使用另一个BiLSTM将匹配向量的两个序列聚合为固定长度向量。  
5. 预测层（Prediction Layer）
该层的目标是做出最终预测。 BiMPM使用两层前馈神经网络来消耗前一层的固定长度矢量，并应用softmax函数获得最终分数。  
![image](https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/基于比较的方法.png)

#### 基于预训练的方法
近年来，随着bert等预训练模型的出现，由于其在大规模的语料库上进行过训练，所以能捕捉到更多的语义信息。近期社区问答领域效果最好的模型通常都采用了基于预训练的方法。这种方法通常将社区问答任务作为一个二分类任务（相似/不相似）来解决，通过[cls]标记将两个句子拼接作为模型的输入，输出为两者为相似的概率。
下面介绍《BERT: Pre-training of Deep Bidirectional Transformers for》中解决社区问答任务的方法:
![image](https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/基于预训练的方法.png)
1. 问题的拼接：  
首先我们将输入送入BERT前，在首部加入[CLS]，在两个句子之间加入[SEP]作为分隔。
2. 相似度计算：  
将输入送入bert后，将得到BERT的输出（句子对的embedding），取[CLS]的向量表示，通过一个线性层，得到两个文档的相似度（相似的概率），最后通过cross-entropy loss来微调模型。

### 3.1. 论文列表
| 会议/年份  | 论文 |链接|
| ------------- | ------------- |------------- |
|CIKM2013   |Learning Deep Structured Semantic Models for Web Search using Clickthrough Data|https://dl.acm.org/doi/10.1145/2505515.2505665 |
|CIKM2016   |A Deep Relevance Matching Model for Ad-hoc Retrieval|https://arxiv.org/abs/1711.08611  |
|KDD2018    |Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction   |https://arxiv.org/pdf/1806.00778.pdf  |
|SIGIR2018  |Sanity Check: A Strong Alignment and Information Retrieval Baseline for Question Answering|https://arxiv.org/pdf/1807.01836 |
|SIGIR2018  |Multihop Attention Networks for Question Answer Matching.|https://dl.acm.org/doi/10.1145/3209978.3210009 |
|SIGIR2018  |Knowledge-aware Attentive Neural Network for Ranking Question Answer Pairs.   |https://dl.acm.org/doi/10.1145/3209978.3210081 |
|WWW2018    |Query Expansion with Neural Question-to-Answer Translation for FAQ-based Question Answering|https://dl.acm.org/citation.cfm?id=3191537 |
|NAACL2018  |Learning to Rank Question-Answer Pairs Using Hierarchical Recurrent Encoder with Latent Topic Clustering|https://www.aclweb.org/anthology/N18-1142 |
|EMNLP2018  |Joint Multitask Learning for Community Question Answering Using Task-Specific Embeddings|https://arxiv.org/abs/1809.08928 |
|ACL2018    |Enhanced LSTM for Natural Language Inference   |https://arxiv.org/pdf/1609.06038.pdf |
|CIKM2019   |A Compare-Aggregate Model with Latent Clustering for Answer Selection|https://arxiv.org/abs/1905.12897|
|WWW2019    |A Hierarchical Attention Retrieval Model for Healthcare Question Answering   |http://dmkd.cs.vt.edu/papers/WWW19.pdf |
|IJCAI2019  |Multiway Attention Networks for Modeling Sentences Pairs | https://www.ijcai.org/Proceedings/2018/0613.pdf |
|AAAI2019   |DRr-Net: Dynamic Re-Read Network for Sentence Semantic Matching | https://www.aaai.org/ojs/index.php/AAAI/article/view/4734/4612 |
|SIGIR2019  |FAQ Retrieval using Query-Question Similarity and BERT-Based Query-Answer Relevance|https://arxiv.org/pdf/1905.02851 |
|SIGIR2019  |Adaptive Multi-Attention Network Incorporating Answer Information for Duplicate Question Detection|http://qizhang.info/paper/sigir2019.duplicatequestiondetection.pdf |
|ACL2019    |Question Condensing Networks for Answer Selection in Community Question Answering|https://www.aclweb.org/anthology/P18-1162.pdf |
|ACL2019    |Simple and Effective Text Matching with Richer Alignment Features|https://www.aclweb.org/anthology/P19-1465/ |
|AAAI2019   |Adversarial Training for Community Question Answer Selection Based on Multi-Scale Matching|https://arxiv.org/abs/1804.08058 |
|WWW2019    |Improved Cross-Lingual Question Retrieval for Community Question Answering|https://dl.acm.org/citation.cfm?id=3313502 |
|NAACL2019  |BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding|https://arxiv.org/abs/1810.04805   |
|NAACL2019  |Alignment over Heterogeneous Embeddings for Question Answering |https://www.aclweb.org/anthology/N19-1274/ |
|ICLR2020   |ALBERT: A Lite BERT for Self-supervised Learning of Language Representations|https://arxiv.org/abs/1909.11942   |
|AAAI2020   |TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection|https://arxiv.org/abs/1911.04118    |



### 3.2. 论文解读
>《DRr-Net: Dynamic Re-read Network for Sentence Semantic Matching》
![image](https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/dr-net.png)

介绍：  
语义匹配一直是一项十分重要的任务，目前，注意力机制大大提升了语义匹配的效果。不过过去的注意力机制通常是一次性关注所有关键词，而人类阅读过程中对关键词的注意往往是变化的。为此，本文提出了一种动态关注关键词的模型  
模型：  
整个模型可以分为三个部分：输入，动态选择关注词，分类
1. 输入（encode）
初始输入为词向量拼接字符级别的词向量以及手工特征（pos，exact match），用一个简单的线性层作一个变换。之后将输入送入一个stack-gru中，即下一层的输入为上一层的输入拼接原始输入（类似残差网络）。
最终，通过一个self-attention将输出的加权和作为句子的表示，文中成为original represtion。
2. 动态重读（ Dynamic Re-read Mechanism）
利用一个注意力机制根据句子的表示，上一次选择的关键词，选择此次的关键词，送入一个gru学习。
3. 分类
对原始表示，重读后的表示，分别拼接表示向量，element-wise的乘积与差，用一个线性层训练。并且动态加权。  

>《Simple and Effective Text Matching with Richer Alignment Features》
![image](https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/re2.png)

介绍：  
提出了一种简单的，不存在复杂特殊结构的文本匹配模型，主要通过point wise信息，上下文相关信息，和前一层提取的相关性信息的结合来表示文档的相关性。  
模型：  
模型为多层类似结构的组合，对两个输入句子采用完全对称的处理。每层的输入为上一层输出与原始embedding的拼接。每一层的输出部分要与上一层的输出相加，每层结构由embedding，encoder和fusion三部分族中。最终将最后一层的输出经过池化后利用predicter部分（一个多层的神经网络）计算最终结果。  
1. embedding部分
采用glove作为初始embeeding，每层的输入为上一层输出与原始embedding的拼接。
2. encoder部分
文中采用了一个cnn作为encoder结构，将encoder输出与encoder的输入拼接，作为每个单词的表示，通过计算内积，得到两个句子attention相关的表示。
3. fusion部分
通过两个句子encoder输出的差，点积和拼接等，通过线性变换得到新的表示。

> 《TANDA: Transfer and Adapt Pre-Trained Transformer Models》
![image](https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/bert.png)

介绍:  
这篇文章主要是通过利用预训练模型来解决答案选择任务。本文提出了一种用于自然语言任务的预训练变换模型精调的有效技术-TANDA( Transfer AND Adapt)。为解决答案选择问题的数据稀缺性问题和精调步骤的不稳定性提供了有效的解决方案。
模型:  
本文的基础模型为Bert模型结构，在经典的任务中，一般只针对目标任务和域进行一次模型精调。对于答案选择任务，训练数据是由问题和答案组成的包含正负标签（答案是否正确回答了问题）的句子对。当训练样本数据较少时，完成答案选择任务的模型稳定性较差，此时在新任务中推广需要大量样本来精调大量的变压器参数。本文提出，将精调过程分为两个步骤：转移到任务，然后适应目标域。首先，使用 AS2 的大型通用数据集完成标准的精调处理。这个步骤应该将语言模型迁移到具体的答案选择任务。由于目标域的特殊性，所得到的模型在目标域的数据上无法达到最佳性能，此时采用第二个精调步骤使分类器适应目标域。





## 4. 参考资料

[2020问答系统（QA）最新论文、书籍、数据集、竞赛、课程资源分析](https://zhuanlan.zhihu.com/p/98688910)  
[Awesome Neural Models for Semantic Match](https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match)  
[《A Review on Deep Learning Techniques Applied to Answer Selection》](https://www.aclweb.org/anthology/C18-1181)  
[用BERT做语义相似度匹配任务：计算相似度的方式](https://www.cnblogs.com/shona/p/12021304.html)  
[自然语言处理中N-Gram模型介绍](https://zhuanlan.zhihu.com/p/32829048)  
[搜索中的权重度量利器: TF-IDF和BM25](https://my.oschina.net/stanleysun/blog/1617727)  
[《基于社区问答的对话式问答系统研究与实现》](https://gb-oversea-cnki-net.e2.buaa.edu.cn/KCMS/detail/detail.aspx?filename=1018813345.nh&dbcode=CMFD&dbname=CMFDREF)

