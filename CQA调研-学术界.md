# CQA调研——学术界

## 目录

  * [1. 任务](#1-任务)
     * [1.1. 背景](#11-背景)
     * [1.2. 任务定义](#12-任务定义)
     * [1.3. 评测标准](#13-评测标准)
     * [1.4. 数据集](#14-数据集)
  * [2. 方法总结](#2-方法总结)
     * [2.1. 基于词频的方法](#21-基于词频的方法)
     * [2.2. 基于语义的方法](#22-基于语义的方法)
        * [2.2.1. 基于表示的方法](#221-基于表示的方法)
        * [2.2.2. 基于交互的方法](#222-基于交互的方法)
     * [2.3. 训练方法](#23-训练方法)
        * [2.3.1 Pointwise方法](#231-Pointwise方法)
        * [2.3.2 Pairwise方法](#232-Pairwise方法)
        * [2.3.3 Listwise方法](#233-Listtwise方法)
  * [3. Paper List](#3-paper-list)
     * [3.1. 论文列表](#31-论文列表)
     * [3.2. 论文解读](#32-论文解读)

## 1. 任务

### 1.1. 背景

#### 智能问答系统发展历史
智能问答系统已经有70年的发展历史。早期的智能问答系统通常只接受特定形式的自然语言问句，而且可以供智能问答系统进行训练的数据也很少，所以无法进行基于大数据的开放领域的问答从而未被广泛使用。进入九十年代之后，由于互联网的发展，大量可供训练的问答对在网上可以被搜集和找到。尤其是 TREC-QA评测的推出，极大推动促进了智能问答系统的发展。目前，已经有很多智能问答系统产品问世。例如IBM研发的智能问答机器人Watson在美国智力竞赛节目《Jeopardy!》中战胜人了选手，其所拥有的DeepQA 系统集成了统计机器学习、信息抽取、知识库集成和知识推理等深层技术。苹果公司的 Siri 系统和微软公司的cortana 分别在 iPhone 手机中和 Windows10 操作系统中都取得了很好的效果。在国内，众多企业和研究团体也推出了很多以智能问答技术为核心的机器人，例如：微软公司的“小冰”、百度公司的“度秘”和中科汇联公司的“爱客服”，我们可以看到，这些机器人不仅提供情感聊天的闲聊功能，而且还能提供私人秘书和智能客服这样的专业功能。这些智能系统的出现标志着智能问答技术正在走向成熟，预计未来还会有更多功能的机器人问世和解决用户的各种需求。 
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/智能问答历史.png)
#### 社区问答
随着网络与信息技术的快速发展，网络上产生了海量信息，人们需要更加快速直接获取有用而准确地信息，推动问答系统的发展。根据系统处理的数据格式，问答系统又可以分为：基于结构化数据的问答系统、基于自由文本数据的问答系统、基于问题答案对数据的问答系统。其中基于问题答案对的问答系统主要分为两种，一种是利用各个公司或者组织在网站上提供的常用问题列表(FAQ)FAQ具有限定领域、质量高、组织好等优点，使得系统回答问题的水平大大提高，但FAQ的获取成本高，这个缺点又制约了基于FAQ的问答系统的应用范围。另一种便是利用问答社区中用户自己产生的问答对的社区问答系统（Community Question Answering, CQA）。随着问答社区的兴起，如 Yahoo!Answers， Quora，StackOverflow，百度知道和知乎等，一种新的问答形式开始大量出现。通过使用社区问答系统，人们不但可以发布问题进行提问以满足自己的信息需求，而且还可以回答其他用户提问的问题来分享自己的知识，让用户所拥有的隐性知识转化成显性知识。

![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/zhihu.png)

### 1.2. 任务定义
**问题-问题匹配**: 
社区问答网站中的问题，通常越来越多是重复问题。检测这些问题有以下几个原因：首先，它会减少冗余; 即如果一个人回答了这个问题一次，他不需要再回答。另外，如果第一个问题有很多答案，并且询问其相似问题，那么答案可以返回给提问者，节省了时间，提升了用户体验。形式化的定义为：  
给定一个问题（后文我们称之为查询）和一个候选问题（后文称为文档）集合,返回根据与查询问题相似性排序的序列  
**问题-答案匹配**：
考虑到社区问答网站接收的流量，在发布的众多答案中找到一个好答案的任务本身就是重要的。这个任务通常被建模为答案选择的任务：  
给定问题q和候选答案集合，然后试着找到最好的候选答案或者每个答案根据与问题相关性排序的列表。候选答案池可能包含也可能不包含多个gold标签。  
由此可见，社区问答的重点问题是计算文本和文本之间的相似性和相关性的问题。

### 1.3. 评测标准
- ACC:判断两个文档是否相似的准确率
- P@1:判断排序第一的答案是否正确
- MAP(Mean Average Precision): 评测整个排序的质量。  
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MAP1.svg)  
其中R表示相关文档的总个数，position(r)表示，结果列表从前往后看，第r个相关文档在列表中的位置。比如，有三个相关文档，位置分别为1、3、6，那么AveP=1/3 * (1/1+2/3+3/6)。
最后，MAP计算所有Query的平均准确率分数：  
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MAP2.svg)  
Q为问题数目总量。
- MRR(Mean Reciprocal Rank)：是把标准答案在被评价系统给出结果中的排序取倒数作为它的准确度，再对所有的问题取平均    
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRR.svg)  

### 1.4. 数据集

#### QQ匹配:
-  [**Quora Question Pairs**](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs):是Quora发布的一个数据集，旨在识别重复的问题。它由Quora上超过40万对问题组成，每个问题对都用0/1标签标注是否为重复问题。  

| Method  | ACC | 论文题目 | 年份 | 论文链接 | code |  
| ------------- | ------------- |------------- |------------- |------------- |------------- |  
|ALBERT|90.5%|ALBERT: A Lite BERT for Self-supervised Learning of Language Representations|2020|https://arxiv.org/pdf/1909.11942v6.pdf |https://github.com/google-research/ALBERT |
|T5-11B|90.4%|Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer|2019|https://arxiv.org/pdf/1910.10683v2.pdf |https://github.com/google-research/text-to-text-transfer-transformer |
|XLNet|90.3%|XLNet: Generalized Autoregressive Pretraining for Language Understanding|2019|https://arxiv.org/pdf/1906.08237v2.pdf |https://github.com/zihangdai/xlnet |


- [**MRPC**](https://www.microsoft.com/en-us/download/details.aspx?id=52398&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F607d14d9-20cd-47e3-85bc-a2f65cd28042%2Fdefault.aspx)：MRPC是Microsoft Research Paraphrase 
Corpus的缩写。它包含从网络新闻源中提取的5,800对句子，以及指示每对是否捕获释义​​/语义对等关系的人工标注。  


| Method  | ACC | 论文题目 | 年份 | 论文链接 | code |
| ------------- | ------------- |------------- |------------- |------------- |------------- |  
|ALBERT|94.0%|ALBERT: A Lite BERT for Self-supervised Learning of Language Representations|2020|https://arxiv.org/pdf/1909.11942v6.pdf |https://github.com/google-research/ALBERT |
|StructBERT|93.9%|StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding|2019|https://arxiv.org/abs/1908.04577 | - |
|ERNIE2.0|93.5%|ERNIE 2.0: A Continual Pre-training Framework for Language Understanding|2019|https://arxiv.org/abs/1907.12412v1   |https://github.com/PaddlePaddle/ERNIE |


- [**LCQMC**](http://icrc.hitsz.edu.cn/info/1037/1146.html):百度发布的一个大型中文问题匹配数据集，数据来自百度知道。每条数据为两个问题和它们的相似性标签（用1/0代表相似/不相似)。  

| Method  | ACC | 论文题目 | 年份 | 论文链接 | code |
| ------------- | ------------- |------------- |------------- |------------- |------------- |  
|ERNIE2.0|87.9%|ERNIE 2.0: A Continual Pre-training Framework for Language Understanding|2019|https://arxiv.org/abs/1907.12412v1   |https://github.com/PaddlePaddle/ERNIE |
|ERNIE1.0|87.4%|ERNIE: Enhanced Representation through Knowledge Integration|https://arxiv.org/abs/1904.09223 |https://github.com/PaddlePaddle/ERNIE |
|BERT|87.0%|BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding|https://arxiv.org/abs/1810.04805 |https://github.com/google-research/bert |


#### QA匹配:
- [**WikiQA**](https://www.microsoft.com/en-us/download/details.aspx?id=52419) :一组公开可用的问题答案对集合，由Microsoft Research收集和注释以用于开放域答案选择问题的研究。    

| Method  | MAP| MRR | 论文题目 | 年份 | 论文链接 | code |
| ------------- | ------------- |------------- |------------- |------------- |------------- |------------|
|TANDA-ROberta|0.920|0.933|TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection|2019|https://arxiv.org/pdf/1911.04118.pdf |https://github.com/alexa/wqa_tanda |
|Comp-Clip + LM + LC|0.764|0.784|A Compare-Aggregate Model with Latent Clustering for Answer Selection|2019|https://paperswithcode.com/paper/a-compare-aggregate-model-with-latent | -|
|RE2|0.7452|0.7618|Simple and Effective Text Matching with Richer Alignment Features|2019|https://www.aclweb.org/anthology/P19-1465/ |https://github.com/alibaba-edu/simple-effective-text-matching |


- [**TRECQA**](https://trec.nist.gov/data/qa.html)从TRECQA8-13的数据中搜集整理的数据集，从每个问题的文档库中自动选择候选答案。该数据集是答案句子选择使用最广泛的基准之一。  

| Method  | MAP| MRR | 论文题目 | 年份 | 论文链接 | code |  
| ------------- | ------------- |------------- |------------- |------------- |------------- |------------| 
|TANDA-ROberta|0.943|0.974|TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection|2019|https://arxiv.org/pdf/1911.04118.pdf |https://github.com/alexa/wqa_tanda |
|BERT-RNN|0.872|0.899|BAS: An Answer Selection Method Using BERT Language Model|https://arxiv.org/ftp/arxiv/papers/1911/1911.01528.pdf |-|
|Comp-Clip + LM + LC|0.868|0.928|A Compare-Aggregate Model with Latent Clustering for Answer Selection|2019|https://paperswithcode.com/paper/a-compare-aggregate-model-with-latent | -|


- [**QNLI**](https://gluebenchmark.com/tasks)：SQuAD数据集的修改版本，允许进行答案选择任务。SQuAD中的上下文段落被分成句子，每个句子都与问题配对。当句子包含答案时，将为问题句子对提供真正的标签。有86,308 / 10,385个问题和428,998 / 169,435个问题/答案对。

| Method  | ACC | 论文题目 | 年份 | 论文链接 | code |
| ------------- | ------------- |------------- |------------- |------------- |------------- |
|ALBERT|99.2%|ALBERT: A Lite BERT for Self-supervised Learning of Language Representations|2020|https://arxiv.org/pdf/1909.11942v6.pdf |https://github.com/google-research/ALBERT |
|Roberta|98.9%|RoBERTa: A Robustly Optimized BERT Pretraining Approach|2019|https://arxiv.org/pdf/1907.11692v1.pdf |https://github.com/huggingface/transformers |
|XLNet|98.6%|XLNet: Generalized Autoregressive Pretraining for Language Understanding|2019|https://arxiv.org/pdf/1906.08237v2.pdf |https://github.com/zihangdai/xlnet |

<!--
### 1.5 Learnboard
| 数据集  | stoa |论文题目|年份|论文链接|code|
| ------------- | ------------- |------------- |------------- |------------- |------------- |
|Quora pairs|90.5(ACC)   |ALBERT: A Lite BERT for Self-supervised Learning of Language Representations   | 2020|https://arxiv.org/pdf/1909.11942v6.pdf |https://github.com/google-research/ALBERT |
|MRPC|91.9(ACC) |StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding|2019|https://arxiv.org/abs/1908.04577 | - |
|LCQMRC|87.9(ACC)|ERNIE 2.0: A Continual Pre-training Framework for Language Understanding|2019|https://arxiv.org/abs/1907.12412v1   |https://github.com/PaddlePaddle/ERNIE |
|Wikiqa|92.0(MAP)|TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection  |2019|https://arxiv.org/pdf/1911.04118.pdf |https://github.com/alexa/wqa_tanda |
|Trecqa|94.3(MAP)|TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection|2019|https://arxiv.org/pdf/1911.04118.pdf |https://github.com/alexa/wqa_tanda |
|QNLI|99.2（ACC)|ALBERT: A Lite BERT for Self-supervised Learning of Language Representations|2020|https://arxiv.org/pdf/1909.11942v6.pdf |https://github.com/google-research/ALBERT |
-->

## 2. 方法总结
可以初步划分为两类，基于词频的方法，通常是一些较为传统的方法，以及基于语义的方法，通常是基于机器学习的方法。

### 2.1. 基于词频的方法
在机器学习出现之前，传统文本匹配方法通常是根据句子中的词频信息进行检索的，如信息检索中的TF-IDF,BM25，语言模型等方法，主要解决字面相似度问题。这些方法由于计算简单，适用范围广，到现在依旧是很多场景下的优秀基准模型。
#### TF-IDF介绍
TF-IDF（term frequency–inverse document frequency）是一种用于资讯检索与文本挖掘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文档集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。  
TF：在一份给定的文件里，词频（term frequency，TF）指的是某一个给定的词语在该文件中出现的次数。对于在某一特定文件里的词语ti来说，它的重要性可表示为：
*TF* = 某个词在文档中的出现次数/文档中的总词数  
*IDF*：逆向文件频率（inverse document frequency，IDF）是一个词语普遍重要性的度量。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到：  
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
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/基于表示的方法.png)

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
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/基于比较的方法.png)

#### 2.2.3 基于预训练的方法
近年来，随着Bert等预训练模型的出现，由于其在大规模的语料库上进行过训练，所以能捕捉到更多的语义信息。近期社区问答领域效果最好的模型通常都采用了基于预训练的方法。这种方法通常将社区问答任务作为一个二分类任务（相似/不相似）来解决，通过[cls]标记将两个句子拼接作为模型的输入，输出为两者为相似的概率。
下面介绍《BERT: Pre-training of Deep Bidirectional Transformers for》中解决社区问答任务的方法:
1. 问题的拼接：  
首先将查询和每一个候选文档一起作为Bert模型的输入，开始加入[CLS]标记。查询和文档之间加入[SEP]标记。利用BPE算法等进行分词，得到Bert模型的输入特征向量。
2. 相似度计算：  
将特征向量输入Bert后，经计算将得到BERT的输出（句子中每个词的向量表示），取[CLS]标记的向量表示，通过一个单层或多层的线性神经网络，得到两个文档的相似度得分（相似的概率)。
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/基于预训练的方法.png)

### 2.3 训练方法
基于语义的方法的训练方法通常可以分为pointwise，pairwise，listwise三种  

#### 2.3.1 Pointwise方法
Pointwise方法是通过近似为回归问题解决排序问题，输入的单条样本为得分-文档，将每个查询-文档对的相关性得分作为标签(Pointwise的由来)，训练模型。预测时候对于指定输入，给出查询-文档对的相关性得分。
#### 特点
其框架具有以下特征：
- 输入中样本是单个文档（和对应查询）构成的特征向量；
- 输出中样本是单个文档（和对应查询）的相关度；
- 假设空间中样本是打分函数；
- 损失函数评估单个文档的预测得分和真实得分之间差异。
该类方法可以进一步分成三类：基于回归的算法、基于分类的算法，基于有序回归的算法。下面详细介绍。
1. 基于回归的算法
此时，输出空间包含的是实值相关度得分。采用传统的回归方法即可。
2. 基于分类的算法
此时，输出空间包含的是无序类别。
对于二分类，SVM、LR 等均可；对于多分类，提升树等均可。
3. 基于有序回归的算法
此时，输出空间包含的是有序类别。通常是找到一个打分函数，然后用一系列阈值对得分进行分割，得到有序类别。采用 PRanking、基于 margin的方法都可以。

#### 缺陷
排序追求的是排序结果，并不要求精确打分，只要有相对打分即可。pointwise类方法并没有考虑同一个查询对应的候选文档间的内部依赖性。一方面，导致输入空间内的样本不是独立同分布的，违反了机器学习的基本假设，另一方面，没有充分利用这种样本间的结构性。其次，当不同查询对应不同数量的docs时，整体loss将会被对应候选文档集合数量大的查询组所支配，前面说过应该每组查询都是等价的。损失函数也没有建模到预测排序中的位置信息。因此，损失函数可能无意的过多强调那些不重要的文档，即那些排序在后面对用户体验影响小的文档。
#### 改进
Pointwise类算法也可以再改进，比如在 loss 中引入基于查询的正则化因子的RankCosine方法。


#### 2.3.2 Pairwise方法
Pairwise方法相较于Pointwise的方法，考虑了文档之间的相对位置关系。输入的单条样本为标签-文档对。对于一次查询的多个结果文档，组合任意两个文档形成文档对作为输入样本。对输入的一对文档对AB（Pairwise的由来），根据A相关性是否比B好，给出结果。对所有文档对进行计算，就可以得到一组偏序关系，从而构造文档全集的排序关系。该类方法的原理是对给定的文档全集S，降低排序中的逆序文档对的个数来降低排序错误，从而达到优化排序结果的目的。
#### 特点
Pairwise类方法，其框架具有以下特征：
- 输入空间中样本是（同一查询对应的）两个文档和对应查询构成的两个特征向量
- 输出空间中样本是样本的相对关系；
- 损失函数评估 doc pair 的预测 preference 和真实 preference 之间差异。
通常来说，Pairwise方法采用margin loss作为优化目标：
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/marginloss.svg)

#### 缺陷
虽然Pairwise方法相较Pointwise方法多考虑了文档对间的相对顺序信息，但还是存在不少问题，许多评测指标考虑到整个排序结果的质量。那么转化成Pairwise时必定会损失掉一些更细粒度的相关度标注信息。
文档对的数量将是候选文档数量的平方，从而 pointwise 类方法就存在的查询间文档数量的不平衡性将在Pairwise 类方法中进一步放大。
Pairwise类方法相对Pointwise方法对噪声标注更敏感，即一个错误标注会引起多个文档对标注错误。
与Pointwise类方法相同，Pairwise类方法也没有考虑同一个查询对应的文档间的内部依赖性，即输入空间内的样本并不是独立同分布的，并且也没有充分利用这种样本间的结构性。
#### 改进
Pairwise类方法也有一些尝试，去一定程度解决上述缺陷，比如：
- Multiple hyperplane ranker，主要针对前述第一个缺陷
- magnitude-preserving ranking，主要针对前述第一个缺陷
- IRSVM，主要针对前述第二个缺陷
- 采用 Sigmoid 进行改进的 pairwise 方法，主要针对前述第三个缺陷
- P-norm push，主要针对前述第四个缺陷
- Ordered weighted average ranking，主要针对前述第四个缺陷
- LambdaRank，主要针对前述第四个缺陷
- Sparse ranker，主要针对前述第四个缺陷

### 2.3.3 Listwise方法
Pointwise类方法将训练集里每一个文档当做一个训练实例，Pairwise类方法将同一个査询的搜索结果里任意两个文档对作为一个训练实例，文档列表方法与上述两种方法都不同，ListWise类方法直接考虑整体序列，针对Ranking评价指标进行优化。比如常用的MAP, NDCG等。
#### 特点
Listwise类方法，其框架具有以下特征：
-输入空间中样本是，同一查询对应的所有文档（与对应的查询构成的多个特征向量（列表）；
-输出空间中样本是这些文档（和对应 query）的相关度排序列表或者排列；
-假设空间中样本是多变量函数，对于文档集合得到其排列，实践中，通常是一个打分函数，根据打分函数对所有docs 的打分进行排序得到文档集合相关度的排列；
-损失函数分成两类，一类是直接和评价指标相关的，还有一类不是直接相关的。
1. 直接基于评价指标的算法
直接取优化排序的评价指标，也算是Listwise类方法中最直观的方法。但这并不简单，因为很多评价指标都是离散不可微的，具体处理方式有这么几种：
1) 优化基于评价指标的ranking error的连续可微的近似，这种方法就可以直接应用已有的优化方法，如SoftRank，ApproximateRank，SmoothRank
2) 优化基于评价指标的 ranking error的连续可微的上界，如 SVM-MAP，SVM-NDCG，PermuRank
3) 使用可以优化非平滑目标函数的优化技术，如 AdaRank，RankGP

2. 非直接基于评价指标的算法
这里，不再使用和评价指标相关的loss来优化模型，而是设计能衡量模型输出与真实排列之间差异的 loss，如此获得的模型在评价指标上也能获得不错的性能。如ListNet，ListMLE，StructRank，BoltzRank等。

#### 缺陷
Listwise 类相较 Pointwise、Pairwise类方法，解决了应该考虑整个排序质量的问题。
listwise 类存在的主要缺陷是：一些排序算法需要基于排列来计算 loss，从而使得训练复杂度较高，如 ListNet和 BoltzRank。此外，位置信息并没有在部分方法的loss中得到充分利用。


### 3.1. 论文列表
| 会议/年份  | 论文 |链接|
| ------------- | ------------- |------------- |
|CIKM2013   |Learning Deep Structured Semantic Models for Web Search using Clickthrough Data|https://dl.acm.org/doi/10.1145/2505515.2505665 |
|CIKM2016   |A Deep Relevance Matching Model for Ad-hoc Retrieval|https://arxiv.org/abs/1711.08611  |
|ACL2017    |Enhanced LSTM for Natural Language Inference   |https://arxiv.org/pdf/1609.06038.pdf |
|KDD2018    |Multi-Cast Attention Networks for Retrieval-based Question Answering and Response Prediction   |https://arxiv.org/pdf/1806.00778.pdf  |
|SIGIR2018  |Sanity Check: A Strong Alignment and Information Retrieval Baseline for Question Answering|https://arxiv.org/pdf/1807.01836 |
|SIGIR2018  |Multihop Attention Networks for Question Answer Matching.|https://dl.acm.org/doi/10.1145/3209978.3210009 |
|SIGIR2018  |Knowledge-aware Attentive Neural Network for Ranking Question Answer Pairs.   |https://dl.acm.org/doi/10.1145/3209978.3210081 |
|WWW2018    |Query Expansion with Neural Question-to-Answer Translation for FAQ-based Question Answering|https://dl.acm.org/citation.cfm?id=3191537 |
|NAACL2018  |Learning to Rank Question-Answer Pairs Using Hierarchical Recurrent Encoder with Latent Topic Clustering|https://www.aclweb.org/anthology/N18-1142 |
|EMNLP2018  |Joint Multitask Learning for Community Question Answering Using Task-Specific Embeddings|https://arxiv.org/abs/1809.08928 |
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

> 《Convolutional Neural Network Architectures for Matching Natural Language Sentences》

*介绍*
匹配模型需要对文本的表示以及它们之间的交互进行建模。之前的模型通常只考虑直接对查询和文档序列进行编码，并且没有考虑查询和文档间的交互作用。本论文针对这两个缺点，提取ARC-I和ARC-II两个模型。前者是基于表示的模型，利用CNN去提取文档特征再计算相似度。后者是基于匹配的模型，首先得到匹配矩阵再用CNN提取特征。
*模型*
1. ARC-I  
模型结构如下图所示
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/ARC-I.jpg)
比较经典的基于表示的匹配模型结构，对于查询和文档分别进行特征提取得到固定维度的向量，而后用MLP进行聚合和分类。因此重点是CNN的用法：
- 运用了多层卷积+pooling的方法
- 卷积操作采用窗口宽度为k1的卷积核，采用宽度为2的max-pooling提取特征，max-pooling可以提取最重要的特征，进而得到查询和文档的表示。
- 单层 CNN 可以捕捉相邻 Term 间得多种组合关系，即local的n-gram 特征。
- 虽然多层 CNN 的堆叠通过感受野的扩张可以得一定的全局信息，但对于序列信息还是不敏感。对语义依赖强的任务效果一般。

2. ARC-II
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/ARC-II.jpg)
ARC-II首先计算查询和文档的单词级别的相似度矩阵，先用1维卷积提取特征，而后用多层二维卷积 + 池化进行计算，最终输入 MLP进行分类。下面介绍卷积层的具体做法：
- 先构建矩阵，假设查询的长度为m，嵌入维度为H，文档长度为n，嵌入维度为H。则矩阵中每个元素是查询中的第i个词向量与 文档中第j个词向量进行拼接得到的向量。因此矩阵的维度是 [m, n, 2H] 。
- 用1维卷积进行扫描。通过这种方式即可以得到查询和文档间的匹配关系，还保留了语义和位置信息。
- 对得到的结果用2维卷积进行处理，池化。池化层的宽度也为2，之后得到最终的表示。

>《DRr-Net: Dynamic Re-read Network for Sentence Semantic Matching》

![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/dr-net.png)
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
![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/re2.png)

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

![image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/bert.png)
*介绍*   
这篇文章主要是通过利用预训练模型来解决答案选择任务。本文提出了一种用于自然语言任务的预训练变换模型精调的有效技术-TANDA( Transfer AND Adapt)。为解决答案选择问题的数据稀缺性问题和精调步骤的不稳定性提供了有效的解决方案。  
*模型*  
本文的基础模型为Bert模型结构，在经典的任务中，一般只针对目标任务和域进行一次模型精调。对于答案选择任务，训练数据是由问题和答案组成的包含正负标签（答案是否正确回答了问题）的句子对。当训练样本数据较少时，完成答案选择任务的模型稳定性较差，此时在新任务中推广需要大量样本来精调大量的变压器参数。本文提出，将精调过程分为两个步骤：转移到任务，然后适应目标域。  
首先，使用 AS2 的大型通用数据集完成标准的精调处理。这个步骤应该将语言模型迁移到具体的答案选择任务。由于目标域的特殊性，所得到的模型在目标域的数据上无法达到最佳性能，此时采用第二个精调步骤使分类器适应目标域。

> 《ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS》

*介绍*：预训练模型通常通过增加模型大小来提升性能。但随着模型的规模越来越大，进一步增加模型大小将带来以下困难：(1)GPU/TPU内存不足(2)训练时间会更长(3)模型退化。  
所以，为了解决上述这些问题，本文提出通过两种参数精简技术来降低内存消耗，并加快BERT的训练速度。此外，本文还引入一个自监督损失(self-supervised loss)，用于对句子连贯性(inter-sentence coherence)建模，并证明该损失函数能够提升多句子作为输入的下游任务的性能。本文所提出的模型ALBERT在 GLUE、RACE 和 SQuAD 这3个基准上都取得了新的SOTA结果，且参数量还少于 BERT-large。  
*模型*  
本文主要提出了两种方法来减少bert模型的参数：
1. 嵌入参数因式分解   ALBERT采用了因式分解的方法来降低参数量，先将词汇表映射到低维参数空间E，再映射到高维参数空间H，使得参数量从O（V* H） 减少到了 O（V * E + E * H），当E远小于H是，能明显减少参数量。
2. 跨层参数共享  
共享encoder中每一层Transformer中的所有参数，之前一般采用只共享全连接层或只共享attention层，ALBERT则更直接全部共享，从实验结果看，全部共享的代价是可以接受的，同时共享权值带来了一定的训练难度，使得模型更鲁棒。  
同时，为了进一步提升 ALBERT 的性能，本文提出一种基于语言连贯性的损失函数SOP（句子次序预测），正例为一篇文档中连续的两个句子，负例为将正例中的两个句子交换顺序。该任务比原始BERT中的NSP（下一句预测）任务更具挑战性。  
基于上述的这3个设计，ALBERT能够扩展为更大的版本，在参数量仍然小于BERT-large的同时，性能可以显著提升。本文在GLUE、SQuAD 和 RACE 这3个自然语言理解基准测试上都刷新了记录：在 RACE 上的准确率为 89.4%，在 GLUE 上的得分为 89.4，在 SQuAD 2.0 上的 F1 得分为 92.2。
！[image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/ALBERT.png)

> 《ERNIE 2.0: A Continual Pre-training Framework for Language Understanding》

*介绍*  
在ERNIE1.0中，通过将BERT中的随机masking改为实体或短语级别（entity or phrase）的masking，使得模型能够从中学习到更多句法语义知识，在许多中文任务上取得SOTA。ERNIE2.0是对ERNIE1.0的一种改进模型，它提出了一种基于持续学习的语义理解预训练框架，使用多任务学习增量式构建预训练任务。ERNIE2.0中，新构建的预训练任务类型可以无缝的加入训练框架，持续的进行语义理解学习。 通过新增的实体预测、句子因果关系判断、文章句子结构重建等语义任务，ERNIE 2.0 语义理解预训练模型从训练数据中获取了词法、句法、语义等多个维度的自然语言信息，极大地增强了通用语义表示能力。
！[image](https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/ERNIE2.0.png)
*模型*  
ERNIE2.0构建了多个预训练任务，试图从 3 个层面去更好的理解训练语料中蕴含的信息：
- Word-aware Tasks: 词汇 (lexical) 级别信息的学习
- Structure-aware Tasks: 语法 (syntactic) 级别信息的学习
- Semantic-aware Tasks: 语义 (semantic) 级别信息的学习
同时，针对不同的 pre-training 任务，ERNIE 2.0 引入了Task Embedding来精细化地建模不同类型的任务。不同的任务用从0 到N的ID表示，每个ID代表了不同的预训练任务。

|任务名称|任务详情|  
| ------------- | ------------- |  
|Knowledge Masking|ERNIE 1.0 中已经引入的 phrase & named entity 知识增强 masking 策略。相较于 sub-word masking, 该策略可以更好的捕捉输入样本局部和全局的语义信息。|
|Capitalization Prediction|针对英文首字母大写词汇（如 Apple）所包含的特殊语义信息,在英文 Pre-training 训练中构造了一个分类任务去学习该词汇是否为大写。|
|Token-Document Relation Prediction|针对一个 segment 中出现的词汇，去预测该词汇是否也在原文档的其他 segments 中出现。|
|Sentence Reordering|针对一个paragraph（包含M个segments），随机打乱segments的顺序，通过一个分类任务去预测打乱的顺序类别|
|Sentence Distance|通过一个 3 分类任务，去判断句对 (sentence pairs) 位置关系 (包含邻近句子、文档内非邻近句子、非同文档内句子 3 种类别)，更好的建模语义相关性。|
|Discourse Relation |通过判断句对 (sentence pairs) 间的修辞关系 (semantic & rhetorical relation)，更好的学习句间语义。|
|IR Relevance|学习 IR 相关性弱监督信息，更好的建模句对相关性。|

## 4. 相关资料

[2020问答系统（QA）最新论文、书籍、数据集、竞赛、课程资源分析](https://zhuanlan.zhihu.com/p/98688910)  
[Awesome Neural Models for Semantic Match](https://github.com/NTMC-Community/awesome-neural-models-for-semantic-match)  
[《A Review on Deep Learning Techniques Applied to Answer Selection》](https://www.aclweb.org/anthology/C18-1181)  
[用BERT做语义相似度匹配任务：计算相似度的方式](https://www.cnblogs.com/shona/p/12021304.html)  
[自然语言处理中N-Gram模型介绍](https://zhuanlan.zhihu.com/p/32829048)  
[搜索中的权重度量利器: TF-IDF和BM25](https://my.oschina.net/stanleysun/blog/1617727)  
[《基于社区问答的对话式问答系统研究与实现》](https://gb-oversea-cnki-net.e2.buaa.edu.cn/KCMS/detail/detail.aspx?filename=1018813345.nh&dbcode=CMFD&dbname=CMFDREF)
[Learning to rank基本算法小结](https://zhuanlan.zhihu.com/p/26539920)
[文献阅读笔记-ALBERT ： A lite BERT for self-supervised learning of language representations](https://blog.csdn.net/ljp1919/article/details/101680220)
[ERNIE及ERNIE 2.0论文笔记](https://www.ramlinbird.com/2019/08/06/ernie%E5%8F%8Aernie-2-0%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0/)
[论文笔记-Multi-cast Attention Networks](https://panxiaoxie.cn/2018/11/04/%E8%AE%BA%E6%96%87%E7%AC%94%E8%AE%B0-Multi-cast-Attention-Networks/)
[文本匹配论文笔记](http://pelhans.com/2019/10/30/text_matching/)
[常见文本相似度计算方法简介](https://zhuanlan.zhihu.com/p/88938220)