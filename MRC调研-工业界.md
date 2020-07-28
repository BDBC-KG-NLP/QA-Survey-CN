# MRC--Machine Reading Comprehension
## 1 任务介绍
### 1.1 简介
- 机器阅读理解是一种利用算法使计算机理解文章语义并回答相关问题的技术，利用人工智能技术为计算机赋予了阅读、分析和归纳文本的能力，本质上是无监督任务。
- 它与英语考试中的阅读理解题目非常相似，阅读一篇英文文章之后，基于此，做后面的几道选择题或者填空题。
<div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9A%E4%B9%89.png"  width=500 alt=机器阅读理解任务样例></div>


### 1.2 定义
- 机器阅读理解基础任务是根据问题，从非结构化文档中寻找合适的答案，因此，研究人员通常将机器阅读理解形式化为一个关于（文档，问题，答案）三元组的监督学习问题。
- 给定一个训练数据集{P，Q，A}，其中，P是文档集，Q是问题集，A是答案集。目标是学习一个函数f：
<div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E6%9C%BA%E5%99%A8%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E5%85%AC%E5%BC%8F.png  width=150 alt=公式></div>



### 1.3 任务类型
- 根据**Answer的类型**，MRC任务可分为4类：**多项选择式、完形填空式、片段抽取式、自由回答式**。
    - **多项选择式（Multiple Choice）**
        - 模型需要从给定的若干选项中选出正确答案。
        - 典型数据集：**MCTest，RACE**
        - 评测指标：**准确率(Accuracy)**
        <div align=center> <img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E5%A4%9A%E9%A1%B9%E9%80%89%E6%8B%A9%E5%BC%8F.png"  width=550 alt=多项选择式></div>


    - **完形填空式（Cloze Test）**
        - 在原文中除去若干关键词，需要模型填入正确单词或短语。
        - 典型数据集：**CNN & Daily Mail、CBT (The Children’s Book Test)、LAMBADA(LAnguage Modeling Boardened to Account for Discourse Aspects)、Who-did-What、CLOTH、CliCR**
        - 评测指标：**准确率(Accuracy)**
        <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E5%AE%8C%E5%BD%A2%E5%A1%AB%E7%A9%BA%E5%BC%8F.png"  width=550 alt=完形填空></div>


    - **片段抽取式（Span Prediction）**
        - 答案限定是文章的一个子句（或片段），需要模型在文章中标明正确的答案起始位置和终止位置。
        - 典型数据集：SQuAD，NewsQA，TriviaQA，DuoRC
        - 评测指标：**精准匹配分数（EM，exact match）**
        <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E5%8C%BA%E9%97%B4%E7%AD%94%E6%A1%88%E5%BC%8F.png"  width=550 alt=片段抽取式></div>
 
 
    - **自由回答式（Free Answering）或总结回答式(Summary of human)**
        - 不限定模型生成答案的形式，允许模型自由生成语句。
        - 典型数据集：**bAbI，MS MARCO（milestone），SearchQA，NarrativeQA，DuReader**
        - 评测指标：**ROUGE-L、BLEU**
        <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E8%87%AA%E7%94%B1%E5%9B%9E%E7%AD%94%E5%BC%8F.png" width=550 alt=自由回答式></div>


> 注：一些数据集设计了“无答案”问题，即一个问题可能在文章中没有合适答案，需要模型输出“无法回答”（unanswerable）。

- **不同任务的对比**
    - 将从五个维度进行比较
        1. 构造数据集难易程度（construction）
        2. 理解和推理程度（understanding）
        3. 答案形式复杂程度(flexibility)
        4. 进行评估的难易程度(evaluation)
        5. 真实应用程度（application）
    <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E5%9B%9B%E7%B1%BBMRC%E4%BB%BB%E5%8A%A1%E5%AF%B9%E6%AF%94.png"  width=650 alt=四类MRC任务对比></div>


MRC任务类型 | 优点 | 缺点
--- | --- | ---
完形填空 | 最容易**构造和评估**数据集<br>(评估 ：将模型答案直接与正确答案比较，并以准确率作为评测标准） | 无法很好地测试对机器的理解，并且与实际应用不符。<br>（原因：因为其答案形式在原始上下文中，仅限于单个单词或名称实体）
多项选择 | 易于**评估**。（将模型答案直接与正确答案比较，并以准确率作为评测标准） | 候选答案导致合成数据集与实际应用之间存在差距。
片段抽取式 | 1 易于构建和评估数据集;<br>（评估： 将模型答案直接与正确答案比较，并以准确率作为评测标准）<br>2 可以以某种方式测试计算机对文本的理解| 答案只能局限在原始上下文的子序列中，与现实应用仍有距离。
自由回答式 | 在理解、灵活性、应用范围方面有优势，最接近实际应用 | 1 构建数据集困难（由于其回答形式灵活）<br>2 有效评估困难



### 1.4 评测方法
- **准确率(Accuracy)**：
    - 表示每个问题回答的准确率。
    - 用于：**完形填空和多项选择**任务
    - 如果m个问题中答对n个，则Accuracy为
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/Accuracy.png  width=200 alt=Accuracy></div>
  
- **问题准确率（Question ACcuracy，QAC）**
    - 含义同准确率，该评测方法名称来自CMRC竞赛

- **篇章准确率（Passage ACcuracy, PAC）**
    - 计算篇章级别的准确率，来自[CMRC竞赛](https://hfl-rc.github.io/cmrc2019/task/)。例如篇章中共有10个空，完全答对所有的空缺才算正确，否则不得分
    - 设共有M篇文章，其中完全答对的篇章数为N，则篇章准确率为
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/PAC.png  width=150 alt=PAC></div>

- **精准匹配分数（EM，exact match）**
    - 用于：**片段抽取式**任务
    - 是准确性的一种变形，可以评估**预测答案片段**是否与**标准真实序列**完全匹配。

- **F1**
    - 同时兼顾**精确率(P)和召回率(R)**。分为微平均F1值（Micro-F1-measure）、宏平均F1值（Macro-F1-measure）
    - 相比于EM，F1分数大致测量了预测值和真实值之间的平均重叠。
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/F1.png  width=250 alt=F1></div>
    
- **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)**
    - 表示评测答案和预测值之间的相似性
    
    - 用于：**自由问答式**任务
    - 公式：设X为m个标记正确答案(ground truth)，Y为n个模型产生的答案，LCS(X,Y)，表示X和Y的最长公共子序列，则
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/Rlcs.png  width=250 alt=ROUGE-L></div>

- **BLEU (Bilingual Evaluation Understudy)**
    - 是一种双语互译质量辅助工具，最初用于衡量翻译性能，表示机器翻译文本与参考文本之间的相似程度
    
    - 用于**自由问答式任务**，表示测量预测值和真实值之间的相似度
    - 具体公式见论文：[Neural Machine Reading Comprehension:Methods and Trends ](https://arxiv.org/pdf/1907.01118v3.pdf)

### 2 数据集
预览：


名称 | 内容 | 类型 | 规模 | 创建者
---|---|---|---|---
bAbI | 英文简短故事 | 自由问答 | 2k | Facebook
MCTest | 英文儿童读物 | 选择题 | 660 | 个虚构故事 | 微软
CNN/DailyMail | 英文新闻 | 完形填空 | CNN：90k文章和380k问题Dailymail：197k文章和879k问题 | DeepMind
RACE | 英语阅读理解 | 选择题  | 2.8w+文章和10w问题 | CMU
HFL-RC | 中文新闻、儿童读物 | 完形填空 | 87w | 哈工大讯飞联合实验室（HFL）
SQuAD 1.0 | 英文维基百科 | 完型填空 | 10w三元组(问题、原文、答案) | Stanford
SQuAD 2.0 | 英文维基百科 | 完型填空 | 500多篇文章，2w多个段落，10w个问题 | Stanford
DuReade | 中文百度搜索和百度知道 | 自由问答 | 30w多个问题，140w个证据文档和660K个人工生成的答案 | 百度
第二届“军事智能机器阅读”挑战赛 数据集 | 中文军事类复杂问题 | 自由问答 | 可下载问答对：train:2.5w / test:0.5w | 
ReCoRD | 英文新闻 | 完形填空 | 12w | 约翰斯·霍普金斯大学、微软研究院
CMRC2019 | 中文新闻、儿童读物 | 完形填空 | 10w段落和100w问题 | 哈工大讯飞联合实验室（HFL）
ChID | 中文新闻、小说、论文 | 完形填空 | 58.1w段落和72.9w个空 | 
法研杯CAIL2019 | 中文裁判文书 | 自由问答 | 约1w | 
CoQA | 英文维基百科、文学、故事、考试、新闻 | 自由问答 | 约12.7w | Stanford

### 2.1 bAbI--Facebook（推理型问答）
- **下载地址**：[链接](https://research.fb.com/downloads/babi/)
- **内容**：人工构造的故事。分为20个评测任务：事实、计数、是否、时间推理、位置推理、基本归纳等
- **答案形式**：通常为一个词或几个词
- **数据量**：共2000个数据，其中1000个训练数据、1000个测试数据


### 2.2 MCTest--微软（选择题）
- **下载地址**：[链接](https://www.microsoft.com/en-us/research/publication/mctest-challenge-dataset-open-domain-machine-comprehension-text/)
- **内容**：真实的英文儿童读物，每篇150-300词，利用众包平台进行标注。
- **数据量**：共有 660 个虚构故事，有两个版本：MCT160和MCT500。要求机器从 4 个候选答案中选择正确的答案。
- **相关论文**：MCTest：A Challenge Dataset for the Open-Domain Machine Comprehension of Test


### 2.3 CNN/DailyMail-DeepMind( 完形填空)
- **下载地址**：[链接](https://github.com/deepmind/rc-data)
- **论文**：《Teaching Machines to Read and Comprehend》
- **内容**：从CNN和Daily Mail上摘取了大量真实新闻语料，然后将每篇文章对应的总结以及复述句子作为问题原型，并从问题中去除某个实体，要求机器能够根据文章内容自动找出答案。要求回答被抽掉的实体，实体在文中出现过
- **数据量**：数据量大。CNN数据集约有90k文章和380k问题，Dailymail数据集有197k文章和879k问题。


### 2.4 RACE-CMU（选择题）
- **下载地**址：[链接](http://www.cs.cmu.edu/~glai1/data/race/)
- **论文**：《RACE：Large-scale Reading Comprehension Dataset From Examinations》
- **内容**
    - 中国中学生英语阅读理解题目，给定一篇文章和 5 道 4 选 1 的题目，包括了 28000+ passages 和 100,000 问题。
    - 规模比MCTest大，且相对CNN&Dailymail和SQuAD，RACE更注重推理能力。
    - 数据以txt格式给出，数据集的high文件夹下有20794篇文章，每篇文章有4个问题；middle文件夹下有7139篇文章，每篇文章有5个问题。
- **数据量**：
    - 训练集：high文件夹下有18728篇文章，占比90%，middle文件夹下有6409篇文章，占比90%；
    - 验证集：high文件夹下有1021篇文章，占比5%，middle文件夹下有368篇文章，占比5%；
    - 测试集：high文件夹下有1045篇文章，占比5%，middle文件夹下有362篇文章，占比5%。



### 2.5 HFL-RC 讯飞和哈工大的中文数据集( 完形填空)
- **下载地址**：[链接](https://github.com/ymcui/Chinese-RC-Dataset)
- **论文**：[《Consensus Attention-based Neural Networks for Chinese Reading Comprehension》](https://arxiv.org/abs/1607.02250)
- **内容**：《人民日报》新闻数据集和《儿童童话》数据集
- **数据量**：数量较大，共87万篇
<div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/HFL-RC%20dataset.png  width=650 alt=HFL-RC dataset.png></div>

### 2.6 SQuAD 1.0-斯坦福（完型填空，答案是变长的，可能是一段话）
- **下载地址**：[链接](https://rajpurkar.github.io/SQuAD-explorer/)
- **论文**：[《SQuAD:100000+ Question for Machine Comprehension of Test》](https://arxiv.org/pdf/1606.05250.pdf)
- **内容**
    - 阅读材料来自536篇英文维基百科，问题和答案主要通过众包的方式，标注人员基于每篇文章，提出最多５个问题并给出对应答案（答案出现在原文中）。目前该数据集共有 536 篇文章、107785 个问答对，答案为原文中的片段。
- **答案形式**：词、短语、句子
- **数据集**：
    - 包含10万个三元组(问题、原文、答案)
    

### 2.7 SQuAD 2.0-斯坦福（完型填空，答案是变长的，可能是一段话）
- **下载地址**：[链接](https://bit.ly/2rDHBgY)

- **论文**：[《Know What You Don't Know: Unanswerable Questions for SQuAD》](https://www.aclweb.org/anthology/P18-2124.pdf)

- **内容**
    -   在原来的 SQuAD（SQuAD 1.1）的十万个问题 - 答案对的基础上，SQuAD 2.0 中新增了超过五万个新增的、由人类众包者对抗性地设计的无法回答的问题。
    -   执行 SQuAD 2.0 阅读理解任务的模型不仅要能够在问题可回答时给出答案，**还要判断哪些问题是阅读文本中没有材料支持的，并拒绝回答这些问题**。如下图中两个问题为不可回答问题，红色字体的答案是错误的。
    <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%20SQuAD2.0%E6%97%A0%E6%B3%95%E5%9B%9E%E7%AD%94%E9%97%AE%E9%A2%98.png" width=450 alt=SQuAD2.0不可回答问题></div>

- **数据量**：数量较大，共500多篇文章，2万多个段落，10万个问题

训练集 | 验证集 | 测试集
--- | --- | --- 
130319个问题 | 11873个问题 | 8862个问题
<div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/SQuAD%202.0%20%E6%95%B0%E6%8D%AE%E9%9B%86.png"  width=450 alt=SQuAD2.0数据集></div>

### 2.8 百度DuReader-2018机器阅读理解技术竞赛（多任务中文数据集）
- **竞赛地址**：[2018机器阅读理解技术竞赛](http://mrc2018.cipsc.org.cn/)
- **下载地址**：[链接](http://ai.baidu.com/broad/download?dataset=dureader)
- **论文**：[《DuReader: a Chinese Machine Reading Comprehension Dataset from Real-world Applications》](https://arxiv.org/abs/1711.05073)
- **内容**：文档和答案从百度搜索和百度知道中获得，答案是由人类回答的，每个问题都对应多个答案
- **任务（问题）类型**：Entity（实体）、Description（描述）和YesNo（是非）
- **答案形式**
    - 实体类问题：答案一般是单一确定的回答，比如：iPhone是哪天发布？
    - 描述类问题：答案一般较长，是多个句子的总结，典型的how/why类型的问题，比如：消防车为什么是红的？
    - 是非类问题：答案为是或者否，比如：39.5度算高烧吗？
- **数据量**：30万多个问题，140万个证据文档和660K个人工生成的答案，每种类型的数据统计

查询类型 | 训练集 | 验证集 | 测试集
--- | --- | --- | ---
总数 | 271574 | 10000 | 20000
Description（描述） | 170428 | 6378 | 12437
Entity（实体） | 76686 | 2825 | 5786
YesNo（是非） | 24460 | 797 | 1777

- **评测方法**：
    - 竞赛基于测试集的人工标注答案，采用ROUGH-L和BLEU4作为评价指标，以ROUGH-L为主评价指标。
- **rank1**：奇点机智Naturali
    - 得分：ROUGE-L 63.38 和 BLEU4 59.23 
    - 技术分享：[链接](https://www.jiqizhixin.com/articles/072801)
    - 模型结构：共四层
        - 特征表示层（Representation)
        - 编码层（Encoding)
        - 匹配层（Matching)：利用Match-LSTM、BiDAF、DCA三种集成模型
        - 答案片段抽取层（Answer Span Extraction)
    <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/2018%E6%9C%BA%E5%99%A8%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E5%A4%A7%E8%B5%9B-%E5%A5%87%E7%82%B9%E6%9C%BA%E6%99%BA%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.png"  width=550 alt=2018机器阅读理解大赛-奇点机智模型结构.png></div>

### 2.9 莱斯杯：全国第二届“军事智能机器阅读”挑战赛 数据集（中文）
- **下载地址**：[链接](https://www.kesci.com/home/competition/5d142d8cbb14e6002c04e14a/content/5)
- **内容**：7万个军事类复杂问题，每个问题对应五篇文章，是目前公开的首个带推理类型的中文机器阅读理解数据集
- **数据量**：
    - 初赛，训练集约2.5万个问题-答案对，测试集（可下载）：包含约0.5万个问题。
    - 复赛，训练集约2.5万个问题-答案对，测试集：包含约1万个问题。
    - 决赛，测试集约0.5万个问题。
- **评价指标**：主要是Rouge-L
  <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E8%8E%B1%E6%96%AF%E6%9D%AF%E6%B5%8B%E8%AF%84%E5%85%AC%E5%BC%8F.png"  width=350 alt=“军事智能机器阅读”挑战赛评价指标></div>

- **rank1**：中科院信工所的向阳而生团队
    - 得分：79.797448
    - 技术点：[技术分享链接](https://www.kesci.com/home/project/5dbbec9f080dc300371eda5d)
        - 阅读器采用基于Bert的roberta-wwm-ext,base中文预训练语言模型
        - 使用了滑窗的方式预处理数据，设计了多答案阅读的损失函数，用来同时学习多个答案。
        - 推理问题，设计了两阶段阅读的方法：提取子问题->回答子问题得到桥接实体->实体替换子问题重构新问题->回答新问题。这一过程，需要两次阅读。
        - ……
    - 具体模型框架如下：
    <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E8%8E%B1%E6%96%AF%E6%9D%AF%EF%BC%9A%E5%85%A8%E5%9B%BD%E7%AC%AC%E4%BA%8C%E5%B1%8A%E2%80%9C%E5%86%9B%E4%BA%8B%E6%99%BA%E8%83%BD%E6%9C%BA%E5%99%A8%E9%98%85%E8%AF%BB%E2%80%9D%E6%8C%91%E6%88%98%E8%B5%9Brank1%E6%A8%A1%E5%9E%8B.png"  width=650 alt=莱斯杯：全国第二届“军事智能机器阅读”挑战赛rank1模型></div>

### 2.10 ReCoRD-约翰斯·霍普金斯大学&微软研究院（完形填空）
- **下载地址**：[链接](https://sheng-z.github.io/ReCoRD-explorer/)
- **论文**：[ReCoRD: Bridging the Gap between Human
and Machine Commonsense Reading Comprehension](https://arxiv.org/pdf/1810.12885.pdf)
- **内容**：包含来自70,000多个新闻文章的120,000多个问题。 与现有的阅读理解数据集不同，ReCoRD包含很大一部分需要常识推理的问题
- **数据量**：总12万问题，其中训练集10万，验证集、测试集各1万
<div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/ReCoRD.%E6%95%B0%E6%8D%AE%E9%9B%86.png"  width=450 alt=ReCoRD数据集></div>

- **评价指标**：精准匹配分数EM、F1
- **rank2**：平安智慧医疗与上海交大
    - 得分：EM 83.09，F1 83.74
    - 论文：[《Pingan Smart Health and SJTU at COIN - Shared Task: utilizing Pre-trained Language Models and Common-sense Knowledge in Machine Reading Tasks》](https://www.aclweb.org/anthology/D19-6011.pdf)
    - 技术：XLNetKGNet、Multi-head Attention、DistMult模型、利用Aho-Corasick算法将段落中的短语与WordNet中的实体进行匹配、KGNet
    - 模型架构
    <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E5%B9%B3%E5%AE%89%E6%99%BA%E6%85%A7%E5%8C%BB%E7%96%97%E4%B8%8E%E4%B8%8A%E6%B5%B7%E4%BA%A4%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%9E%B6.png"  width=550 alt=平安智慧医疗与上海交大模型架构></div>

### 2.11 CMRC2019：第三届“讯⻜杯”中⽂机器阅读理解评测（句⼦级完形填空）
- **下载地址**：[链接](https://github.com/ymcui/cmrc2019)
- **论文地址**：[A Sentence Cloze Dataset for Chinese Machine Reading Comprehension](https://arxiv.org/abs/2004.03116)
- **答案类型**：句子
- **数据量**：1w文档和10w问题
<div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/CMRC2019dataset.png"  width=550 alt=CMRC2019dataset></div>

- **评价指标**
    - 问题准确率（Question ACcuracy, QAC）：计算问题（一个空即一个问题）级别的准确率
    <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/CMRC-QAC.png"  width=180 alt=CMRC-QAC></div>
    
    - 篇章准确率（Passage ACcuracy, PAC）：计算篇章级别的准确率
        - 例如：篇章中共有10个空，完全答对所有的空缺才算正确，否则不得分。
    <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/CMRC-PAC.png"  width=180 alt=CMRC-PAC></div>
    
- **rank1**：平安金融壹账通
    - 竞赛报告：[《平安金融壹账通gammalab团队CMRC2019竞赛报告》](https://hfl-rc.github.io/cmrc2019/resource/report.zip)

    - 技术
        - **预训练**
            - BERT模型优化，预训练语料丰富化（采集了百科，新闻，知乎等多源的数据重新训练了BERT）
            - Sentence Insertion和全词mask任务
            - 句子篇章关系预测任务
            - 预训练模型的领域迁移
        - **数据增强**
            - 简单负样本增强
            - 动态数据增强
        - **数据处理**   
            - SentencePiece字词混合模型
            - 动态预测
            <div align=center><img src="https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E5%B9%B3%E5%AE%89%E9%87%91%E8%9E%8D%E5%A3%B9%E8%B4%A6%E9%80%9Asentence-insertion.png"  width=650 alt=平安金融壹账通sentence-insertion></div>



### 2.12 ChID数据集-成语阅读理解大赛
- **下载地址**：[链接](https://www.biendata.xyz/competition/idiom/data/)
- **内容**
    - 数据集的语料来源于论文ChID: A Large-scale Chinese IDiom Dataset for Cloze Test（ACL 2019）。
    - 基于选词填空的任务形式，提供大规模的成语填空训练语料。
    - 在给定若干段文本下，选手需要在提供的候选项中，依次选出填入文本中的空格处最恰当的成语。
- **数据量**：58.1w段落和72.9w个填空
- **评测方法**：填空正确率
  <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E6%88%90%E8%AF%AD%E7%AB%9E%E8%B5%9B%E8%AF%84%E6%B5%8B%E6%96%B9%E6%B3%95-%E5%A1%AB%E7%A9%BA%E6%AD%A3%E7%A1%AE%E7%8E%87.png  width=180 alt=成语阅读理解大赛评测指标正确率></div>
  
- **rank1**：wssb
    - 得分：90.97823
    - 代码：[链接](https://www.biendata.xyz/models/category/2762/L_notebook/)
    - 技术：
    - 模型：

### 2.13 法研杯CAIL2019（自由回答）
- **下载地址**：[链接](https://github.com/china-ai-law-challenge/CAIL2019)
- **论文**：[《CJRC: A Reliable Human-Annotated Benchmark DataSet for Chinese Judicial Reading Comprehension》](https://arxiv.org/abs/1912.09156)

- **内容**
    - 数据内容来自中国法官文书网，主要涉及民事和刑事的一审判决书，总共约1万份数据。属于篇章片段抽取型阅读理解比赛
- **数据量**
    - 数据集共包括268万刑法法律文书，共涉及183条罪名，202条法条，刑期长短包括0-25年、无期、死刑。
- **评测方法**：
    - 罪名预测任务：微平均F1值（Micro-F1-measure）
    - 法条推荐任务：宏平均F1值（Macro-F1-measure）
    - 刑期预测任务：预测出的刑期与案件标准刑期之间的差值距离
- **rank1**:
    - 技术分享：[《法研杯2019阅读理解赛道冠军方案分享（含PPT）》](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411097&idx=1&sn=b6988b9e3ac5c2d4fd5b8ff4e92e2065&utm_source=tuicool&utm_medium=referral)
    - 策略：
        - 1 在bert-base-chinese上，基于大赛数据做fine-tune
        - 2 是否类问题的解决方案：tf-idf 方式，对比原问题与段落中的每一句来寻找答案相关段落，从而对模型进行训练
        - 3 数据增强
        - 4 阈值调整：解决不平衡数据集问题
    - 模型
        - 将google的bert输出接上词性等特征加上一层传统的highway与GRU后通过MLP来判断答案的label与Span的位置
        <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/CAIL2019-rank1%20model.jpg  width=650 alt=CAIL2019-rank1 model.jpg></div>
        
### 2.14 CoQA stanford(自由回答)
- **下载地址**：[链接](https://stanfordnlp.github.io/coqa/)

- **论文**：[《CoQA: A Conversational Question Answering Challenge》](https://arxiv.org/abs/1808.07042)
- **内容**
    - 数据来自儿童故事、文学、初高中英语考试、新闻、维基百科、Reddit和科学等七个不同的领域文章，包含127,000多个问题，并从8000多个对话中收集了答案。每次对话都是通过将两名群众工作者配对以问题和答案的形式聊聊一段段落而收集的。
    - CoQA的独特功能包括：1）问题是对话性的；2）答案可以是自由格式的文本；3）每个答案还带有一个在段落中突出显示的证据子序列；和4）段落是从七个不同的领域收集的。
- **数据量**：包含12.7万多个问题
- **评测方法**：官方评估脚本
- **rank1**：追一科技
    - 论文：[《Technical report on Conversational Question Answering》](https://arxiv.org/abs/1909.10772)
    - 技术分享：[追一科技CoQA冠军方案分享：基于对抗训练和知识蒸馏的机器阅读理解方案](https://www.leiphone.com/news/201911/g5eqn6CjbLPI5GDU.html)

    - 策略：
        - 1) Baseline模型：RoBERTa。原因：RoBERTa在语言模型预训练的过程中用了更多领域的语料
        - 2) 增加依据标注任务，并同步进行多任务训练。
        - 3) 采用对抗训练和知识蒸馏等训练方法
        <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/COQA%20rank1%20%E8%BF%BD%E4%B8%80%E7%A7%91%E6%8A%80%E4%BE%9D%E6%8D%AE%E6%A0%87%E6%B3%A8%E4%BB%BB%E5%8A%A1.png  width=500 alt=COQA rank1 追一科技依据标注任务></div>



## 3 模型介绍
- 近年来模型的总体框架图
<div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%80%BB%E4%BD%93%E6%A1%86%E6%9E%B6%E5%9B%BE.jpeg  width=650 alt=近年来模型的总体框架图></div>

### 3.1 BIDAF--Allen AI
- **论文**：[Bidirectional Attention Flow for Machine Comprehension ](https://arxiv.org/pdf/1611.01603.pdf)
- **简介**
    - 模型利用双向注意力流(Bi-Directional Attention Flow，BiDAF)得到了一个问题感知的上下文表征。问题感知的上下文表征是所给段落和问题之间的交互。
    - 创新：双向注意力机制，它在QA任务中充当编码器或者推理单元中的一环对后续的性能产生更大的影响。
- **特点**
    - vector：BiDAF并不是将文本总结为一个固定长度的vector，而是将vector流动起来，以便减少早期信息加权和的损失
    - Memory-less：在每一个时刻，仅仅对 query 和当前时刻的 context paragraph 进行计算，并不直接依赖上一时刻的 attention，这使得后面的 attention 计算不会受到之前错误的 attention 信息的影响
    - 一层的interaction，只有文章和问题的相关性：计算了 query-to-context(Q2C)和 context-to-query(C2Q)两个方向的 attention 信息，认为 C2Q 和 Q2C 实际上能够相互补充。
- **模型架构**
    - BiDAF共有6层，依次是
        - Character Embedding Layer
        - Word Embedding Layer
        - Contextual Embedding Layer
        - Attention Flow Layer
        - Modeling Layer
        - Output Layer
    - 其中前三层是一个多层级上下文不同粒度的表征编码器。第四层则是双向注意流层，这是原文的核心层。第五层是一个编码层，编码第四层输出的问题感知的上下文表征。第六层就是一个预测答案的范围。
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/BIDAF.png  width=650 alt=BIDAF></div>
- **模型效果**
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/BIDAF%E6%95%88%E6%9E%9C.png  width=450 alt=BIDAF模型效果></div>

- **解读文章**：[BiDAF：机器理解之双向注意力流 ](https://zhuanlan.zhihu.com/p/53470020)

### 3.2 R-NET--微软
- **论文**：[R-NET: Machine Reading Comprehension with Self-matching Networks ](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)

- **简介**：
    - 基于自匹配网络的机器阅读理解
    - R-NET是首个在某些指标中接近人类的深度学习模型。
- **创新**：
    - 两层的interaction，与BIDAF相比新提出self-matching的注意力机制通过将文本自身进行匹配来优化表示，从而有效地对整个段落中的信息进行编码。
- **模型架构**
    - 用于阅读理解和问题回答的端到端神经网络模型，由以下四部分组成：
    - 1）**Question & Passage Encoding**
        - 多层的双向循环神经网络编码器，用于为问题和文本建立表示	
    - 2）**Question-Passage Matching**
        - 门控匹配层（gated matching layer），用于匹配问题和文本
        - 这一层将问题中的向量和文本中的向量做一个比对，这样就能找出那些问题和哪些文字部分比较接近。
    - 3）**Passage Self-Matching**
        - 自匹配层（self-matching layer），用于整合整个段落的信息
        - 将问题和文本的匹配结果放在全局中进行比对。这些都是通过注意力机制（attention）达到的。
    - 4）**Answer Prediction**
        - 基于答案边界预测层的提示网络（pointer-network）
        - 针对挑出的答案候选区中的每一个词汇进行预测，哪一个词是答案的开始，到哪个词是答案的结束。
        <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/R-net.jpg  width=650 alt=R-net></div>
- **模型效果**
    - 1 在SQuaAD数据集上的结果
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/R-Net%E6%95%88%E6%9E%9C.png  width=650 alt=R-net在SQuaAD数据集上的结果></div>
    - 2 在MS-MARCO数据集上的结果
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/R-Net%E5%9C%A8MS-MARCO%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E7%9A%84%E7%BB%93%E6%9E%9C.png  width=650 alt=R-net在MS-MARCO数据集上的结果></div>
    
- **解读文章**：[R-NET机器阅读理解（原理解析）](https://zhuanlan.zhihu.com/p/36855204)

### 3.3 QANET-- Google
- **论文**：[QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension ](https://openreview.net/pdf?id=B14TlG-RW)

- **简介**
    - 之前RNN模型的缺点是**train/test的时候速度都很慢，不利于实时实时处理**。因此作者提出了一种新的不需要RNN结构的QA模型-QANet，在很大程度改进了QA模型的速度。
    
    - 这个模型用**卷积负责学习局部特征，而self-attention负责全局特征**。相比之前的模型它的速度更快，且在更长时间下能够得到更好的结果。
    
- **创新**
    - 在普通阅读理解模型的embedding和modeling encoder layer中**只用卷积和self-attention而没有用RNN**(比只用self-attention↑2.7F1)。
    
    - 这种做法的好处是，
        - 可以并行处理输入数据，使得模型的速度大大加快；
        - 可以利用cnn已经成熟的regularization方法(比如layer dropout、stochastic depth)等(↑0.2F1)。
        
- **亮点**：速度快。训练速度和推理速度都比之前提高很多
- **模型架构**：模型共有5层，依次是
    1. **Input Embedding Layer**
    2. **Embedding Encoder Layer**
        - 主要由三部分组成: [convolution_layer  + self-attention_layer + feed-forward_layer]
        
        - 用的是占空间更小的depthwise separable convolution
        - self-attention部分采用了multi-head attention mechanism。这种attention对input(query和keys)的每个位置计算所有位置加权和，算query和key的点积来衡量他们的相似度
    
    3. **Context-Query Attention Layer**
        - 计算context和query之间的联系，找出其中的关键词语
    
    4. **Model Encoder Layer**
        - 利用convolution+attention从整体上考虑context和query之间的关系。
    
    5. **Output Layer**
        - loss中使用了了context中的一个词是answer的首词和尾词的概率
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/QANET.png  width=650 alt=QANET></div>
    
- **模型效果**
    > 实验过程中使用多倍的数据增强方式（回译法--Backtranslation），对EM和F1都有很大贡献。作者主要通过两个翻译模型来进行数据增强：一个将英语翻译成其他语言(1个句子变成k个句子，论文中k=5)，另一个将它所得的结果再翻译回英语(变成k*k个句子)
 
    1. 准确率
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/QANet-acc.png  width=400 alt=QANET准确率></div>
    
    2. 速度
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/QANet-speed.png  width=650 alt=QANET速度></div>

### 3.4 GPT & BERT
- **简介**
    - 句子级别信息的迁移，整体使用transformer结构 
    - 改变的是输入数据流：问题+分隔符+文章的顺序序列
    - 迁移学习：特征维度的迁移


## 4  产品案例
### 4.1 SMRC 阅读理解工具包--搜狗
- **软件文档**：[Sogou Machine Reading Comprehension Toolkit](https://github.com/sogou/SogouMRCToolkit/blob/master/README.md)
- **简介**
    - SMRC（Sogou Machine Reading Comprehension）是TensorFlow 版本的阅读理解工具集合，从相关数据集的下载到最后模型的训练和测试都非常全面。
- **模型介绍**：
    - 分解为4个步骤：数据集读取、预处理、模型构建、训练和评估，对每步都进行了抽象和模块化，以简洁的接口呈现。
        1. **数据集读取模块**（dataset_reader）
            - 该模块集成了对SQuAD 1.0/2.0、CoQA以及中文数据集CMRC的读取和预处理功能。
        2. **数据预处理**（data、utils）
            - data部分包含词表构建模块和负责特征变换和数据流的batch生成器。utils用于提取语言学特征。
        3. **模型构建**（nn、models）
            - nn（神经网络）由机器阅读理解中的常用组件组成，可以快速构建和训练原型模型，避免部分重复工作。model中集成了常见的机器理解模型，如双向注意力流BiDAF、DrQA、融合网络FusionNet、QANet等等。
        4. **模型训练与评估**（examples）
            - 这一部分是运行不同模型的示例
        <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/SMRCarchitecture.png  width=650 alt=SMRC architecture></div>

### 4.2 SLQA -- 阿里
- **全称**：Semantic Learning for Question Answering
- **应用场景**：
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E9%98%BF%E9%87%8C%E5%B0%8F%E8%9C%9C/%E5%88%86%E9%A2%86%E5%9F%9F%E7%9A%84%E6%A8%A1%E5%9E%8B%E6%94%AF%E6%8C%81.jpg  width=650 alt=分领域的模型支持></div>
    
    - **阿里小蜜**：把机器阅读理解应用在大规模客服场景下
    - **店小蜜**：通过机器阅读理解技术，让机器对详情页中的商品描述文本进行更为智能的阅读和回答，在降低卖家服务成本的同时提高购买转化率。
    - **企业小蜜**：税务法规解读场景
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E9%98%BF%E9%87%8C%E5%B0%8F%E8%9C%9C/%E7%A8%8E%E5%8A%A1%E6%B3%95%E8%A7%84%E8%A7%A3%E8%AF%BB%E5%9C%BA%E6%99%AF.jpg  width=650 alt=税务法规解读场景></div>
- **技术分享**：[链接](https://myslide.cn/slides/6148?vertical=1)
- **模型结构**
    - 基于分层融合注意力机制的深度神经网络系统
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E9%98%BF%E9%87%8C%E5%B0%8F%E8%9C%9C/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E4%B8%9A%E5%8A%A1%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.jpg  width=650 alt=阅读理解业务模型结构></div>
    
    - SLQA 系统包含如下基本结构：Encoder Layer（文本表征），Attention Layer（注意力机制），Match Layer（问题篇章匹配）以及 Output Layer（答案预测）。
        1. **Encoder Layer**
            - **问题及篇章中词向量表示**。
            - 采用了**多层双向 LSTM捕捉语序间的依赖**，并分别对篇章和问题进行主题和重点词关注。
            
        2. **Attention Layer** （重点探索和研究工作）
            - **对齐问题和篇章，语义相似度计算**。每一次对齐都基于下层信息并在此基础上更加细化（paragraph→sentence→phrase→word），采用的方式分别为 Co-Attention（篇章到问题，问题到篇章），Self-Attention（问题自身，篇章自身）。
            
            - 引进**注意力机制**，带着问题找答案
        3. **Match Layer（或Modeling  Layer）** 
            - 用于做**融合信息后的问题和篇章匹配**，团队采用双线性矩阵来学习经过多层信息过滤后的篇章和问题匹配参数，由于在前一阶段无关信息已经被过滤，最后的匹配可完成答案的定位工作。
        4. **Output Layer** 
            - **基于问题和篇章匹配预测答案位置**。结合匹配信息对篇章中词汇进行标注，预测相应词汇是答案开始位置或结束位置的概率。之后，模型会抽取可能性最高的一段连续文本作为答案。
- **基于阅读理解的问答处理流程**
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E9%98%BF%E9%87%8C%E5%B0%8F%E8%9C%9C/%E5%9F%BA%E4%BA%8E%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E7%9A%84%E9%97%AE%E9%A2%98%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B1-2.jpg  width=650 alt=基于阅读理解的问题处理流程1&2></div>
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E9%98%BF%E9%87%8C%E5%B0%8F%E8%9C%9C/%E5%9F%BA%E4%BA%8E%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3%E7%9A%84%E9%97%AE%E9%A2%98%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B3-4.jpg  width=650 alt=基于阅读理解的问题处理流3&4></div>
    
- **线上模型效果**
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/%E9%98%BF%E9%87%8C%E5%B0%8F%E8%9C%9C/%E7%BA%BF%E4%B8%8A%E6%A8%A1%E5%9E%8B%E6%95%88%E6%9E%9C.jpg  width=650 alt=线上模型效果></div>

### 4.3 Z-Reader -- 追一科技
- **简介**
    - 该模型效果获得 CMRC 2018阅读理解比赛第一名
    - 已进行商业应用的尝试，如开发**直接帮顾客从文档集中找到答案的阅读理解机器人**。
- **技术分享**：[视频：阅读理解进阶三部曲——关键知识、模型性能提升、产品化落地](https://mooc.yanxishe.com/course/596/learn?lessonid=2928#lesson/2928)
- **模型描述**
    - Embedding
        - 由四部分组成：ELMO、pos embedding、query type embedding、word match
        <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/Z-Reader%20Embedding.png  width=350 alt=Z-Reader Embedding></div>
    - Encoding
        - Bidirectional GRU, multi-layers
    - Attention
        - Extra gated-dropout for query
    - Prediction
        - pointer network
        - prob = start * stop
    - Training
        - Born-Again Neural Networks 再生神经网络：利用知识蒸馏收敛到更优的模型
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/Z-Reader.png  width=550 alt=Z-Reader></div>
    

### 4.4 阅读理解(RC)模型-AmazonQA
- **背景**
    - 每天都有成千上万的顾客针对亚马逊页面上的产品提出问题。如果他们幸运的话，一段时间过后会有一位知识渊博的客户回答他们的问题。
    - **基于评论的问答任务**：由于许多问题可以根据已有产品的评价进行回答，为此提出了基于评论的QA任务。
- **方法描述**
    - 给定一个评论语料库和一个问题，QA系统自动综合一个答案。
    - 亚马逊引入了一个新的数据集，并提出了一种结合信息检索技术来选择相关评论(给定问题)和“阅读理解”模型来综合答案(给定问题和评论)的方法。
    - 数据集中的问题、段落和答案都是从真实的人类交互中提取的。
- **系统概念图**
    <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/AmazonQA%E7%B3%BB%E7%BB%9F%E6%A6%82%E5%BF%B5%E5%9B%BE.png  width=550 alt=AmazonQA系统概念图></div>

- **代码**：[链接](https://github.com/amazonqa/amazonqa)
- **模型描述**
    - Baseline Models
        1. Language Models
        <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/MRC/AmazonQA%20LM.png  width=550 alt=AmazonQ LM></div>
        2. Span-based QA Model：Span-based QA Model
            

## 5 其他领域应用
- 传统的搜索引擎只能返回与用户查询相关的文档，而阅读理解模型可以在文档中精确定位问题的答案，从而提高用户体验。
1. **搜索引擎、广告、推荐**
    - 基于MRC可以完成知识抽取、QA等重要的NLP任务，可用于搜索引擎、广告、推荐
2. **智能法律**
    - 用于自动处理和应用各种错综复杂的法律法规实现对案例的自动审判，这正可以利用机器阅读理解在处理和分析大规模文档方面的速度优势。
3. **智能教育**
    - 利用计算机辅助人类的学习过程。机器阅读理解在这个领域的典型应用是作文自动批阅。自动作文批阅模型可以作为学生写作时的助手，理解作文语义，自动修改语法错误，个性化总结易错知识点。这些技术与当前流行的在线教育结合，很有可能在不久的将来对教育行业产生颠覆性的影响。
4. **客户服务**
    - 利用机器阅读理解在产品文档中找到与用户描述问题相关的部分并给出详细解决方案，可以大大提高客服效率。
5. **智能医疗**
    - 阅读理解模型能根据患者症状描述自动查阅大量病历和医学论文，找到可能的病因并输出诊疗方案。

## 总结
1. 应用最多的模型是R-Net和BIDAF
1. 工业界应用大多用到了注意力机制

## 参考文献
- [【NLP】详聊NLP中的阅读理解（MRC）](https://blog.csdn.net/hacker_long/article/details/104604146)
- [搜狗开源最新NLP研究成果，打造业内最全机器阅读理解工具包SMRC](https://blog.csdn.net/yH0VLDe8VG8ep9VGe/article/details/90709172)
- [【重磅】机器阅读理解终于超越人类水平！权威竞赛排名中国霸榜，阿里、MSRA、腾讯前二](https://developer.aliyun.com/article/363415)
- [机器是如何“阅读理解”的？| NLP基础](https://cloud.tencent.com/developer/article/1582788)
- [机器阅读理解是什么？有哪些应用？终于有人讲明白了](https://my.oschina.net/u/4497340/blog/4300671)
- [机器阅读理解打破人类记录，解读阿里iDST SLQA 技术](https://blog.csdn.net/Uwr44UOuQcNsUQb60zk2/article/details/79060596?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.nonecase)
- [阿里小蜜机器阅读理解技术揭秘-张佶](https://myslide.cn/slides/6148?vertical=1)
- [机器阅读理解方向-前沿跟进资源整理](https://zhuanlan.zhihu.com/p/128453331)
- [MRC综述: Neural MRC: Methods and TrendsNeural Machine Reading Comprehension: Methods and Trends](https://arxiv.org/pdf/1907.01118v3.pdf)
- [机器阅读理解中你需要知道的几个经典数据集](https://www.imooc.com/article/24766)
- [DuReader：百度大规模的中文机器阅读理解数据集](https://www.imooc.com/article/24845)
- [从短句到长文，计算机如何学习阅读理解-微软研究院](https://www.msra.cn/zh-cn/news/features/machine-text-comprehension-20170508)
- [法研杯2019阅读理解赛道冠军方案分享（含PPT）](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411097&idx=1&sn=b6988b9e3ac5c2d4fd5b8ff4e92e2065&utm_source=tuicool&utm_medium=referral)
- [追一科技CoQA冠军方案分享：基于对抗训练和知识蒸馏的机器阅读理解方案](https://www.leiphone.com/news/201911/g5eqn6CjbLPI5GDU.html)
- [BiDAF：机器理解之双向注意力流](https://zhuanlan.zhihu.com/p/53470020)
- [[论文笔记]QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension ](https://qianqianqiao.github.io/2018/10/14/new/)
