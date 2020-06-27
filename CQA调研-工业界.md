CQA--工业界

## 目录
  * [1 任务](#1-任务)
    * [1.1 任务定义](#11-任务定义)
    * [1.2 数据集](#12-数据集)
  * [2 方法及模型](#2-方法及模型)
    * [2.2 用于问答匹配的方法](#22-用于问答匹配的方法)
       * 2.2.1 规则匹配(又称“句式法”)
       * 2.2.2 深度学习多分类模型（CNN\DNN\LSTM\…）
       * 2.2.3 基于Siamese networks神经网络架构
       * 2.2.4 Interaction-based networks
    * [2.3 用于Chatbot的方法](#23-用于Chatbot的方法)
    * [2.4 用于机器阅读理解的方法](#24-用于机器阅读理解的方法)
    * [2.5 用于跨领域迁移学习方法](#25-用于跨领域迁移学习方法)
  * [3 产品举例](#3-产品举例)
  * [4 问题难点及未来研究的方向](#4-问题难点及未来研究的方向)
  * [5 相关资料](#5-相关资料)

## 1 任务

### 1.1 任务定义
**C**ommunity **Q**uestion **A**nswer，中文名称是社区问答。是利用半结构化的数据（问答对形式）来回答用户的提问，其流程通常可以分为三部分。

1. 问题解析，对用户输入的问题进行分词，纠错等预处理步骤。

2. 召回部分，利用信息检索引擎如Lucence等根据处理后的问题提取可能的候选问题。

3. 排序部分，利用信息检索模型对召回的候选问题进行相似度排序，寻找到最相似的问题并返回给用户。

   

### 1.2 任务分类

通常，根据应用场景的不同，可以将CQA任务分为两类：
- FAQ问答: 在智能客服的业务场景中，对于用户频繁会问到的业务知识类问题的自动解答（以下简称为FAQ）是一个非常关键的需求，可以说是智能客服最为核心的用户场景，可以最为显著地降低人工客服的数量与成本。这个场景中，知识通常是封闭的，而且变化较为缓慢，通常可以利用已有的客服回复记录提取出高质量的问答对作为知识库。
- 社区问答: 问答对来自于社区论坛中用户的提问和回答，较为容易获取，但是相对质量较低。而且通常是面向开放域的，知识变化与更新速度较快。

### 1.3 评测标准
- 查全率：用以评价系统对于潜在答案寻找的全面程度。例如：在回答的前30%中保证一定出现正确答案。
- 查准率：即准确率，top n个答案包含正确答案的概率。这一项与学术界一致。
- 问题解决率：与具体业务和应用场景紧密相关
- 用户满意度/答案满意度：一般对答案满意度的评价方式是在每一次交互后都设置一个评价，客户可以对每一次回答进行评价，评价该答案是否满意。但是这样的评价方式容易让客户厌烦，因为客户是来解决问题的，不是来评价知识库里面的答案是否该优化。
- 问题识别率/应答准确率：指智能客服机器人正确识别出客户的问题数量在所有问题数中的占比。目前业内评价智能机器人比较常用的指标之一。
- 问题预判准确率：指用户进入咨询后，智能客服机器人会对客户可能咨询的问题进行预判。如京东的问题预判，是通过其长期数据积累和模型给每个用户添加各种标签，可以提供更个性化和人性化的服务。例如，京东JIMI了解用户的性别、情绪类型、近期购买历史等。当用户开始交流时，就会猜到他可能要询问一个关于母婴商品的使用方法或是一个售后单的退款情况，这就是问题预判。如果预判准确的话，只需在几次甚至一次的交互中获得智能客服机器人专业的问题解答，从而缩短客户咨询时长。
- 意图识别准确率：要想解答用户的问题，机器人首先需要结合上下文环境，从用户提问中准确识别用户咨询的意图是什么，然后返回对应的答案。
- 拦截率：机器人代替人工解决的用户咨询比例
- 24H未转人工率：指客户咨询了智能机器人后的24H内是否有咨询人工客服

### 1.4 难点
- 有标记的相似文本训练数据标注难以自动获取
- 高质量的问答对数据获取与维护成本较高
- 用户可能的输入类型较多，匹配模型的鲁棒性无法保证

### 1.5  数据集



##### “技术需求”与“技术成果”项目之间关联度计算模型（需求与成果匹配)  

- **比赛链接**：https://www.datafountain.cn/competitions/359

- **任务目标**
  
  - 根据项目信息的文本含义，为供需双方提供关联度较高的对应信息（需求——成果智能匹配
  
- **数据来源**
  - 数据来自中国·河南开放创新暨跨国技术转移大会云服务平台（www.nttzzc.com）
  - **人工标注关联度的方法**：从事技术转移工作的专职工作人员，阅读技术需求文本和技术成果文本，根据个人经验予以标注。关联度分为四个层级：强相关、较强相关、弱相关、无相关。

- **数据具体说明**：https://www.datafountain.cn/competitions/359/datasets

- **评价指标**：使用MAE系数
  - 平均绝对差值是用来衡量模型预测结果对标准结果的接近程度一种衡量方法.MAE的值越小，说明预测数据与真实数据越接近。

  <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/CQA-industry-MAE.png  width=200 alt=MAE公式></div>
  <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/CQA-industry-score.png  width=180 alt=最终结果></div>
  - 最终结果越接近1分数越高。
  
- **top1方案及结果**
  - 解决方案：https://www.sohu.com/a/363245873_787107 
  
  - 主要利用数据清洗、数据增广、孪生BERT模型
  
  - 最高得分：0.80285150
  
    
  

##### cMedQA2 （医疗问答匹配）
- **比赛链接**：https://www.mdpi.com/2076-3417/7/8/767
- **数据来源**
   - 寻医寻药网站中的提问和回答， 数据集做过匿名处理
- **数据分布**
  - 总量有108,000个问题，203,569个答案
      - 训练集中有100,000个问题，188,490个答案
      - 验证集有4,000个问题，有7527个答案
      - 测试集有4,000个问题，有7552个答案。
- **top1 解决方案**：[Multi-Scale Attentive Interaction Networks for Chinese Medical Question Answer Selection](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8548603)



##### 智能客服问题相似度算法设计——第三届魔镜杯大赛
- **比赛链接**：https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=1
- **任务目标**
  
    - 计算客户提出问题与知识库问题的相似度
- **数据来源**
  
    - 智能客服聊天机器人真实数据
- **数据分布及描述**
  
    - https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=1
- **评价指标**：logloss，logloss分数越低越好
- **方案及结果**
    - [rank6方法](https://qrfaction.github.io/2018/07/25/%E9%AD%94%E9%95%9C%E6%9D%AF%E6%AF%94%E8%B5%9B%E7%AD%94%E8%BE%A9PPT/)(rank6结果0.145129，top1结果0.142658)
      
      - 主要利用传统特征(如最长公共子序列、编辑距离等)，结构特征(构造图结构。将q_id作为node，(qi,qj)作为edge，得到一个单种边的同构图，然后计算qi,qj的公共边权重和等结构)，还有半监督、相似传递性、早停优化等
      
        


##### CCKS 2018 微众银行智能客服问句匹配大赛
- **比赛链接**：https://biendata.com/competition/CCKS2018_3/
- **任务目标** 
  - 针对中文的真实客服语料，进行问句意图匹配
- **数据来源**
  - 所有语料来自原始的银行领域智能客服日志，并经过了筛选和人工的意图匹配标注。
- **数据具体说明**：https://biendata.com/competition/CCKS2018_3/data/
- **评价指标**：Precision、Recall、F1值、ACC
- **top1评测论文**：An Enhanced ESIM Model for Sentence Pair Matching with Self-Attentionhttp://ceur-ws.org/Vol-2242/paper09.pdf?crazycache=1



##### AFQMC 蚂蚁金融语义相似度
- **比赛链接**：https://dc.cloud.alipay.com/index?click_from=MAIL&_bdType=acafbbbiahdahhadhiih#/topic/intro?id=3
- **任务目标**
  - 给定客服里用户描述的两句话，用算法来判断是否表示了相同的语义
- **数据来源**
  - 所有数据均来自蚂蚁金服金融大脑的实际应用场景。
- **数据分布**
  - 初赛阶段提供10万对的标注数据作为训练数据，包括同义对和不同义对，可下载；复赛阶段不提供下载
  - 具体说明：[链接](https://dc.cloud.alipay.com/index?click_from=MAIL&_bdType=acafbbbiahdahhadhiih#/topic/data?id=3)
- **评测指标**：F1-score为准（得分相同时，参照accuracy排序）
- **top1解决方案**：[链接](https://www.jiqizhixin.com/articles/2018-10-15-14 )。
    - 主要利用char-level feature、ESIM 模型、ensemble



##### OPPO手机搜索排序query-title语义匹配数据集
- **比赛链接**：[OGeek算法挑战赛--实时搜索场景下搜索结果ctr预估](https://tianchi.aliyun.com/competition/entrance/231688/introduction)
- **数据集链接**：https://pan.baidu.com/s/1Hg2Hubsn3GEuu4gubbHCzw (密码7p3n)
- **数据来源**
  
  - 该数据集来自于OPPO手机搜索排序优化实时搜索场景, 该场景就是在用户不断输入过程中，实时返回查询结果。 该数据集在此基础上做了相应的简化， 提供了一个query-title语义匹配。
- **数据分布**
  - 初赛数据约235万 训练集200万，验证集5万，A榜测试集5万，B榜测试集25万
  - 具体说明：https://tianchi.aliyun.com/competition/entrance/231688/information
- **评测指标**：F1 score 指标，正样本为1
- **top1 解决方案**
  - 答辩链接：[链接](https://tianchi.aliyun.com/course/video?spm=5176.12586971.1001.83.1770262auKlrTZ&liveId=41001)(00:42开始)
  
  - 主要应用：数据预处理、CTR问题的特征挖掘、TextCNN&TF-IDF、attention net、数据增强、回归CTR模型融合lightGBM、阈值选择。（rank 1,rank2两只队伍都是使用了lightGBM模型和模型融合）
  
  - 最后得分：0.7502
  
    


##### 医疗问题相似度衡量竞赛数据集（医疗问题匹配、意图匹配）
- **比赛链接**：[中国健康信息处理会议举办的医疗问题相似度 衡量竞赛](https://biendata.com/competition/chip2018/) 

- **任务目标**：针对中文的真实患者健康咨询语料，进行问句意图匹配。给定两个语句，要求判定两者意图是否相同或者相近

- **数据来源**
  - 来源于真实问答语料库，该任务更加接近于智能医疗助手等自然语言处理任务的实际需求
  - 所有语料来自互联网上患者真实的问题，并经过了筛选和人工的意图匹配标注。
  
- **数据分布**
  - 训练集包含20000条左右标注好的数据（经过脱敏处理，包含标点符号），供参赛人员进行训练和测试。
  - 测试集包含10000条左右无label的数据（经过脱敏处理，包含标点符号）
  - 具体描述：[链接](https://biendata.com/competition/chip2018/data/)
  
- **评测指标**：Precision，Recall和F1值。最终排名以F1值为基准

  


## 2 方法及模型

### 2.1 无监督方法

#### 2.1.1 规则匹配(又称“句式法”)

  - 优点：可控、高效、易于实现
  - 目前，很多机器人都有规则匹配的部分。
  - **具体做法**：针对FAQ库中的标问和相似问进行分词、提炼出大量的概念，并将上述概念组合，构成大量的句式，句式再进行组合形成标问。
  > 例如，标问“华为mate30现在的价格是多少？”，拆出来“华为mate30”是cellphone概念，“价格是多少”是askMoney概念，“现在”是time概念，那么“华为mate30现在的价格是多少？”就是cellphone+askMoney+time。用户输入"华为mate30现在卖多少钱？"进行分词，可以得到相同的句式和概念组合，就能够命中“华为mate30现在的价格是多少？”这个相似问了。

在拥有较大数据量积累的场景，一般采用有监督的深度神经网络，可以解析文本并抽取高层语义。

### 2.1.3 无监督文本表示

在缺少标记数据的场景，我们可以利用算法对文本本身进行表示，再利用常用的向量距离计算方法（如WSD，余弦距离，欧式距离等）进行相似性度量。常见的无监督文本表示方法主要可以分为两种，一种是基于词频信息的方法，一种是基于词向量的方法。

- 基于词频信息的方法：传统的文本表示方法通常是基于词频特征的，例如TF-IDF，语言模型等。

  - TF-IDF：将文档表示为其每个单词的TF-IDF值向量形式，并通过计算两个文本向量表示的余弦相似度来衡量其相似性。

  - 语言模型：根据现有的文本对每个单词由一篇文档生成的概率根据词频进行建模，将一段文本由另一段文本生成的概率作为其相似度得分。

  - <div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/language model.png  width=650 alt=语言模型></div>

- 基于浅层语义的方法，如LSA，LDA等。

  - LSA 

    LSA在潜在语义分析之上引入了主题概念。它是一种语义含义，对文档的主题建模不再是矩阵分解，而是概率分布（比如多项式分布），这样就能解决多义词的分布问题，并且主题是有明确含义的。但这种分析的基础仍然是文档和词的共现频率，分析的目标是建立词/文档与这些潜在主题的关系，而这种潜在主题进而成为语义关联的一种桥梁。

  - LDA

    如果说pLSA是频度学派代表，那LDA就是贝叶斯学派代表。LDA通过引入Dirichlet分布作为多项式共轭先验，在数学上完整解释了一个文档生成过程，其概率图模型如图所示。

    

    和pLSA概率图模型不太一样，LDA概率图模型引入了两个随机变量α和β，它们就是控制参数分布的分布，即文档-主题符合多项式分布。这个多项式分布的产生受Dirichlet先验分布控制，可以使用变分期望最大化（Variational EM）和吉布斯采样（Gibbs Sampling）来推导参数，这里不展开叙述。

    总体来讲，主题模型引入了“Topic”这个有物理含义的概念，并且模型通过共现信息能学到同义、多义、语义相关等信息。得到的主题概率分布作为表示，变得更加合理有意义。有了文档的表示，在匹配时，我们不仅可以使用之前的度量方式，还可以引入KL等度量分布的公式，这在文本匹配领域应用很多。当然，主题模型会存在一些问题，比如对短文本推断效果不好、训练参数多速度慢、引入随机过程建模避免主题数目人工设定不合理问题等。随着研究进一步发展，这些问题基本都有较好解决，比如针对训练速度慢的问题，从LDA到SparseLDA、AliasLDA, 再到LightLDA、WarpLDA等，采样速度从O(K)降低O(1)到。

  

- 基于词向量的方法： word embedding技术如word2vec，glove等已经广泛应用于NLP，极大地推动了NLP的发展。既然词可以embedding，句子也可以。该类算法通常是基于词袋模型的算法，如TF-IDF加权平均，SIF等。

  - SIF

### 2.1.4 领域迁徙

迁移学习的模型有两类，一种是unsupervised，另外一种是supervised。前者假设完全没有目标领域的标注数据，后者假设仅有少部分目标领域的标注数据。在实际的商业应用中主要以supervised的迁移学习技术为主，同时结合深度神经网络（DNN）。在这个设定下主要有两种框架：

- Fully-Shared Model:用于比较相似的两个领域。
- Specific-Shared Model: 用于相差较大的两个领域。

## 2.2 有监督匹配算法

#### 2.2.1 基于意图识别的算法

- 问答匹配任务在大多数情况下可以转化为二分类或多分类任务。
- 神经网络中会有两大输入，左边N会输入结构化数据，比如个人属性以及浏览操作历史纪录，右边V会输入一些非结构化数据，比如前几轮问的问题和序列，对于这些非结构化的数据我们会有句子编码器解析这些数据，当需要考虑到句子的语序关系的时候会使用CNN或者RNN网络结构；上层的话，会结合用户的Embedding和句子的Embedding去输出。
- 工业真正的场景中，用户问题的问题个数是不固定的，所以会把最后一层Softmax更改为多个二分类模型。模型图如下：

<div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/多个二分类模型.jpeg  width=650 alt=多个二分类模型模型图></div>

#### 2.2.2深度文本匹配模型

一般来说，深度文本匹配模型分为两种类型，表示型和交互型。


#### 表示型模型
表示型模型更侧重对表示层的构建，它首先将两个文本表示成固定长度的向量，之后计算两个文本向量的距离来衡量其相似度。这种模型的问题是没有考虑到两个句子词级别的关联性。容易失去语义交代呢。

##### Siamese networks模型

- Siamese networks(孪生神经网络)是一种相似性度量方法，内部采用深度语义匹配模型（DSSM，Deep Structured Semantic Model），该方法在检索场景下使用点击数据来训练语义层次的匹配。
- Siamese networks有两个输入(Input1 and Input2),将两个输入feed进入两个神经网络(Network1 and Network2)，这两个神经网络分别将输入映射到新的空间，形成输入在新的空间中的表示。通过Loss的计算，评价两个输入的相似度。
- 基于Siamese networks神经网络架构，比如有Siamese结构的LSTM、CNN和ESIM等。


##### DSSM 模型
- **论文**：[Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)
- **模型简介**
    - 先把 query 和 document 转换成 BOW 向量形式，然后通过 word hashing 变换做降维得到相对低维的向量，feed给 MLP 网络，输出层对应的低维向量就是 query 和 document 的语义向量（假定为 Q 和 D）。计算(D, Q)的余弦相似度后，用 softmax 做归一化得到的概率值是整个模型的最终输出，该值作为监督信号进行有监督训练。
- **模型结构**：

<div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/CQA-industry-DSSM.png  width=650 alt=DSSM></div>

##### Siamese LSTM 模型
- 解读链接：https://zhuanlan.zhihu.com/p/48188731
- **模型简介**
>- RNN 模型提出之前，比较两段文本的相似性都习惯用词袋模型或者 TF-IDF 模型，但没有用到上下文的信息，而且词与词之间联系不紧密，词袋模型难以泛化。
>- LSTM 或者 RNN 模型可以去适应变成的句子，比如通过 RNN 可以将两个长度不同的句子 encode 成一个相同长度的语义向量，这个语义向量包含了各自句子的语义信息，可以直接用来比较相似性。
> - Siamese Recurrent Architectures 就是将两个不一样长的句子，分别 encode 成相同长度的向量，以此来比较两个句子的相似性。

- **论文**：[Siamese Recurrent Architectures for Learning Sentence Similarity](http://people.csail.mit.edu/jonasmueller/info/MuellerThyagarajan_AAAI16.pdf)

- **模型结构**
    - 1.**问题的语义表示向量抽取**： 通过LSTM完成，在问题1和问题2的对称网络中，这部分LSTM共享权重。
    - 2.**语义向量相似性计算**：计算语义表示向量的平方距离和角度，再喂给多层感知机MLP进行分类。
    - 模型如下所示
    
<div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/CQA-industry-siamese%20LSTM.png    width=500 alt=CQA-industry-siamese LSTM 模型></div>

#### 2.1.4 交互型模型

交互型模型认为全局的匹配度依赖于局部的匹配度，在输入层就进行词语间的先匹配，之后利用单词级别的匹配结果进行全局的匹配。它的优势是可以很好的把握语义焦点，对上下文重要性合理建模。由于模型效果显著，业界都在逐渐尝试交互型的方法。
##### ESIM （Enhanced LSTM）
- **论文**：Enhanced LSTM for Natural Language Inference
- **源码**：[链接](https://github.com/coetaur0/ESIM)
- **模型结构**
    - 由输入编码，局部推理模型和推断合成三部分构成。
    - 模型使用了双向LSTM，并引入attention机制。
    - 模型如下所示
<div align=center><img src=https://github.com/BDBC-KG-NLP/QA-Survey/blob/master/image/CQA-industry-ESIM.png  width=400 alt=ESIM></div>
  - 同时对问题和答案进行特征加权的Attention方案
    



#### Bimpm(Bilateral Multi-perspective Matching) 

- 论文： Bilateral Multi-Perspective Matching for Natural Language Sentences
- 源码： [链接](https://github.com/zhiguowang/BiMPM)
- 模型简介
- **模型结构**
  - 

## 3 产品案例
### 产品1: YiBot
**简介**
- YiBot是由深圳追一科技有限公司自主研发，应用目前最前沿的自然语言处理及深度学习算法，为企业级客户提供的一套智能客服机器人系统。

**FAQ问题优化**
> YiBot的FAQ问题优化是从用户问题和智能客服机器人的回答出发，合理拆分、合并已有FAQ，并优化问句和答案；同时采用问句聚类技术，挖掘新知识点，补充更新已有知识点，淘汰废弃知识点，形成一个正向循环，不断优化知识结构，提高拦截率。
- **FAQ发现**
> 将用户问句进行聚类，对比已有的FAQ，发现并补足未覆盖的知识点。将FAQ与知识点一一对应。
- **FAQ的拆分与合并**
> FAQ拆分是当一个FAQ里包含多个意图或者说多种情况的时候，YiBot后台会自动分析触达率较高的FAQ，聚类FAQ对应的问句，按照意图将其拆分开来。
- **FAQ合并**
> 最终希望希望用户的每一个意图能对应到唯一的FAQ，这样用户每次提问的时候，系统就可以根据这个意图对应的FAQ直接给出答案。而如果两个FAQ意思过于相近，那么当用户问到相关问题时，就不会出现一个直接的回答，而是两个意图相关的推荐问题，这样用户就要再进行一步选择操作。这时候YiBot就会在后台同样是分析触达率较高的FAQ，分析哪一些问句总是被推荐相同的答案，将问句对应的意图合并。
- **淘汰机制**
> 分析历史日志，采用淘汰机制淘汰废弃知识点，如已下线业务知识点等。

**FAQ答案优化**
- **挖掘对话，进行答案优化**

> 如果机器人已经正确识别意图但最后仍然转人工，说明知识库的答案不对，需要进一步修正这一类知识点相对应的答案。
- **分析头部场景，回答应用文本、图片、自动化解决方案等多元化方式**
> 比如在电商场景中，经常会有查询发货到货时间、订单状态等的场景。利用图示指引、具体订单处理等方式让用户操作更便捷。

### 产品2: [百度AnyQ--ANswer Your Questions](https://github.com/baidu/AnyQ)

**简介**

- AnyQ开源项目主要包含面向FAQ集合的问答系统框架、文本语义匹配工具SimNet。

**系统框架**

- AnyQ系统框架主要由Question Analysis、Retrieval、Matching、Re-Rank等部分组成。
- 框架中包含的功能均通过插件形式加入，如Analysis中的中文切词，Retrieval中的倒排索引、语义索引，Matching中的Jaccard特征、SimNet语义匹配特征，当前共开放了20+种插件。

**特色**

- **特色1 框架设计灵活，插件功能丰富，有助于开发者快速构建、快速定制适用于特定业务场景的 FAQ 系统**
> AnyQ 系统集成了检索和匹配的丰富插件，通过配置的方式生效；以相似度计算为例，包括字面匹配相似度 Cosine、Jaccard、BM25 等，同时包含了语义匹配相似度。且所有功能都是通过插件形式加入，用户自定义插件，只需实现对应的接口即可，如 Question 分析方法、检索方式、匹配相似度、排序方式等。
- **特色2 极速语义检索**
> 语义检索技术将用户问题和 FAQ 集合的相似问题通过深度神经网络映射到语义表示空间的临近位置，检索时，通过高速向量索引技术对相似问题进行检索。
- **特色3 业界领先语义匹配技术 SimNet**
> AnyQ 使用 SimNet 语义匹配模型构建文本语义相似度，克服了传统基于字面匹配方法的局限，增强 AnyQ 系统的语义检索和语义匹配能力。
- **其他**：针对无任何训练数据的开发者，AnyQ 还包含了基于百度海量数据训练的语义匹配模型，开发者可零成本直接使用。

### 产品3: [腾讯知文--结构化FAQ问答引擎](https://cloud.tencent.com/developer/article/1172017  )
基于结构化的FAQ的问答引擎流程由两条技术路线来解决
- 无监督学习，基于快速检索
- 有监督的学习，基于深度匹配

**无监督的快速检索方法**

采用了三个层次的方法来实现快速检索的方法
- **层次1：基础的TFIDF提取query的关键词，用BM25来计算query和FAQ库中问题的相似度**。这是典型的词汇统计的方法，该方法可以对rare word比较鲁棒，但同时也存在词汇匹配缺失的问题。
- **层次2：采用了language model（简写LM）的方法**。主要使用的是Jelinek-Mercer平滑法和Dirichlet平滑法，对于上面的词汇匹配问题表现良好，但是也存在平滑敏感的问题。
- **层次3：最后一层使用Embedding，采用了LSA/word2vec和腾讯知文自己提出的Weighted Sum/WMD方法**，以此来表示语义层面的近似，但是也同样引发了歧义问题。

**监督的深度匹配方法**

采用了两条思路
- **思路1 基于Siamese networks神经网络架构**。这是一种相似性度量方法，内部采用深度语义匹配模型（DSMM，Deep Structured Semantic Model），该方法在检索场景下使用点击数据来训练语义层次的匹配
- **思路2 Interaction-based networks，同时对问题和答案进行特征加权的Attention方案**。

### 产品4: [阿里小蜜](https://www.alixiaomi.com/#/)
**为什么要做阿里小蜜？**

阿里小蜜出现之前团队发现的问题有2个，第一个是需要对话机器人的业务很多，第二点是独立开发者的开发成本又很高。为了解决这两个问题，团队需要做一套平台产品来赋能开发者。如果做一个平台能够提供一些非常易于操作的开发工具，有丰富的内置能力，有强大的 AI 算法能力，以及全生命周期的配套工具，那么这些独立的开发者或者企业就能够做到零代码开发，快速交付具有鲁棒对话的机器人，并且该机器人可以在线上进行持续迭代优化。

**为什么阿里小蜜是新一代的对话开发平台？**

第一个是新的设计思路：将原来以 Intent 为中心的设计思路转变为以 Dialog 为中心的设计思路。第二个是新的开发模式：从原先表单开发方式转变为可视化拖拽式开发。第三个是 End-to-End AI-Powered：从原来只提供单点的 NLU、DM 模型 AI 赋能方式转变为 End-to-End AI-Powered 平台，在整个生命周期的各个阶段都会用 AI 来赋能开发者。基于这样的想法，阿里的团队打造了对话工厂（Dialog Studio）这个产品，属于小蜜机器人解决方案中的一个核心基础能力，该产品已经嵌入到小蜜家族（阿里小蜜，店小蜜，钉钉小蜜，云小蜜等）的各个平台中，支持了包括阿里电商，电信，政务，金融，教育等领域的各个场景的对话机器人业务。这就是为什么要做一个平台型对话机器人产品的初衷。

<div align=center><img src=https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/640-4.jpeg width=650 alt=ESIM></div>

- 下面针对这三点进行具体介绍：

  - 1. 从 Intent 为中心到以 Dialog 为中心
以 Intent 为中心的方式每次要创建单个的意图，如果遇到一个复杂场景，需要创建很多个意图堆积在一起，每次开发只能看到当前正在创建的意图，因为该技术起源于原来的 Slot Filling 方式，只能解决简单的，Slot Filling 这种任务形式对话，任务场景比较受限。而以 Dialog 为中心的设计思路，把人机对话过程通过图的方式展现出来，对话步骤用图上的节点进行抽象，开发者在设计对话流的时候，这种方式能提供一个全局的视野，而且图的节点抽象比较底层，可以进行各种任务的配置，适用场景比较广泛。

<div align=center><img src=https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/640-3.jpeg width=650 alt=ESIM></div>

  - 2. 从表单式开发到可视化拖拽式开发
在开发模式上将原来的表单式开发方式变成了可视化拖拽式开发方式。原来表单式开发方式以 Intent 为中心，所以对于开发者来说更像做一道一道表单填空题，只能单点控制，整个对话流程非常不直观，所有 Intent 压缩在一个表单，填写复杂。在可视化拖拽方式中，整个对话流过程的每一个节点都可以通过简单拖拽方式进行完整描述，拥有全局视野，整体可控。

<div align=center><img src=https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/640-2.jpeg width=650 alt=ESIM></div>

  - 3. 从单点模型到全生命周期 AI-Powered

第三点是从原来单点 NLU、DM 模型到全生命周期 AI-Powered，在对话机器人开发的各个阶段都利用了 AI 算法能力赋能开发者，加速开发过程和降低开发成本。在设计阶段尝试了半自动对话流设计辅助，让开发者从冷启动阶段就能够设置出第一版的对话流。在对话流的构建阶段，我们推出了智能荐句功能，也就是说在编写用户话术的时候，机器人可以进行话术推荐和联想。在测试阶段，我们推出了机器人诊断机器人功能，可以大大减少测试的工作量，增速测试过程。在在线服务阶段，会有全套的 AI-Powered 对话引擎，包括 NLU、DM 等算法。在数据回流阶段，通过 Active Learning 将日志数据进行自动标注，用于后续模型迭代训练。在持续学习阶段，我们构建了一套完整自动模型训练、评测、发布 Pipeline，自动提升线上机器人效果。

<div align=center><img src=https://github.com/BDBC-KG-NLP/CQA-Survey/blob/master/images/640.jpeg width=650 alt=ESIM></div>

**阿里小蜜的核心算法之一：自然语言理解(NLU)方法**

- 无样本冷启动方法：写一套简单易懂的规则表示语法
- 小样本方法：我们先整理出一个大数量级的数据，每一个类目几十条数据，为它建立 meta-learning 任务。对于一个具体任务来说：构建支撑集和预测集，通过 few-shot learning 的方法训练出 model，同时与预测集的 query 进行比较，计算 loss 并更新参数，然后不断迭代让其收敛。这只是一个 meta-learning 任务，我们可以反复抽样获得一系列这样的任务，不断优化同一个模型。在线预测阶段，用户标注的少量样本就是支撑集，将 query 输入模型获得分类结果。模型的神经网络结构分为3部分，首先是 Encoder 将句子变成句子向量，然后再通过 Induction Network 变成类向量，最后通过 Relation Network 计算向量距离，输出最终的结果。具体地，Induction Network中把样本向量抽象到类向量的部分，采用 matrix transformation 的方法，转换后类边界更清晰，更利于下游 relation 的计算。在 Induction Network 的基础上，又可以引入了 memory 机制，形成Memory-based Induction Network ，目的是模仿人类的记忆和类比能力，在效果上又有进一步提升。
- 多样本方法：构建一个三层的模型，最底层是具有较强迁移能力的通用模型 BERT，在此基础上构建不同行业的模型，最后用相对较少的企业数据来训练模型。这样构建出来的企业的 NLU 分类模型，F1 基本都在90%+。性能方面，因为模型的结构比较复杂，在线预测的延时比较长，因此通过知识蒸馏的方法来进行模型压缩，在效果相当的同时预测效率更快了。

## 4 问题难点及未来研究的方向

## 5 相关资料

- [阿里小蜜新一代智能对话开发平台技术解析](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247494321&idx=1&sn=7f58bafd7f1962e17f3162ef0917c431&chksm=fbd758ddcca0d1cb19c452c40697c816f788d29b90af4f703a0fc776897f80b087d0a3bc885a&scene=27#wechat_redirect)
- [阿里云小蜜对话机器人背后的核心算法](https://mp.weixin.qq.com/s/ksVbQq42ay5lxcfqNwBgxA)
- [阿里小蜜：智能服务技术实践及场景探索](https://mp.weixin.qq.com/s/uzmcISuDbf7EkralufAKhA)
- [干货 | 阿里小蜜-电商领域的智能助理技术实践](https://mp.weixin.qq.com/s/eFm89Q_AMeYFTrJl4uLOgA)
- [阿里小蜜机器阅读理解技术揭秘](https://myslide.cn/slides/6148#)
- [从学术前沿到工业领先：解密阿里小蜜机器阅读的实践之路](https://zhuanlan.zhihu.com/p/62217668)
- [云知声：深度文本匹配在智能客服中的应用](https://www.jiqizhixin.com/articles/2018-10-23-15)
- [[NLP点滴——文本相似度](https://www.cnblogs.com/xlturing/p/6136690.html)](https://www.cnblogs.com/xlturing/p/6136690.html#simhash)
