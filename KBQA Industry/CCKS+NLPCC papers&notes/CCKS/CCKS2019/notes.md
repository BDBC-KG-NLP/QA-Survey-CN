# CCKS 2019 论文笔记

## 混合语义相似度的中文知识图谱问答系统

### 结果

- F1：73.54%

### 系统

![1586349819279](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019/pictures/1586349819279.png)

#### 指称识别

- 子串匹配：生成问题全部子串，剪枝（长度>=2，指称不能被完全包含）
- 命名实体识别召回人名指称。
- 启发式方法识别指称。针对可以被其它指称包含的指称，把该实体的一度关系召回，与问题进行匹配，匹配成功的留下该指称。

#### 实体链接

- 实体与问题匹配特征
  - 实体名称与问题的匹配度
  - 实体二度子图与问题的匹配度
  - 实体类型与问题的匹配度
  - 采用集合距离/word2vec
- 流行度特征
  - 实体在图谱出现频率
  - 实体不同的一度关系个数
- 指称重要度特征
  - 指称是否被引号或书名号包含
  - 指称是否在开头或结尾
  - 指称和疑问词的距离
  - 指称是否包含数字或字母
  - ...
  - 基于lambdarank的排序算法

#### 模板匹配组件

- 召回每个实体的二度子图
- 剪枝一：当实体流行度过ths，慢删除该节点的关联边。
- 剪枝二：某些路径的方向未在训练集中出现，删除这种路径
- 三种模板

#### 路径排序组件

- 39个特征
- 路径与问题字面匹配特征：jaccard，编辑距离
- 路径与问题的语义匹配特征：bert答案类型特征
- 答案类型匹配特征
- 实体链接的概率
- 候选路径自身特征(匹配哪类模板)



## Combining Neural Network Models with Rules for Chinese Knowledge Base Question Answering

### 结果

F1：73.075

### 模型

![1586395493614](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019/pictures/1586395493614.png)

#### path similarity matching

![1586396085552](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019/pictures/1586396085552.png)

- Bert sequence labeling + string matching to do ner
- bert-trained classifier + rules -》 multi-limit questions, one-hop questions, chain questions and difficult-to-categorize questions
- generate candidate paths for different templates （这个具体没细说）， use bert similarity model to obtain the optimal path
- 实现：分类采用bert+classification
- 实现：单跳：语义匹配采用bert+匹配 二分类，1：100正负样例比，多跳全部当做chain-questions做(s1-r1-r2-ANS)，1：50正负样例比

#### relationship similarity matching

![1586397182767](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019/pictures/1586397182767.png)

- classify the question (one hop / multi hop), 类似ner 标注 entity
- entity linking based on a set of features(没细说), get top1 entity
- 计算图谱中每个实体的关系和问题的相似度，获取top1 relation
- 针对chain question (s1-r1-o1-r2-ANS)获取o1后再进行一次查找
- 针对multihop，将所有可能的ent和rel同时找出来，然后在数据库里面找交集
- 模型：
  - mention entity model: bert+bilstm+crf
  - entity linkling model (xgboost)
    - order, mention score, questions and mention char matching, questions and mention char matching, questions and entity semantic similarity, questions and entity char matching, maximum similarity between questions and entity character matching
  - relation extraction model：same as path similarity matching， 只不过是问题和relation拼接，正负样例1：5

#### rule-based method

- 将问题分割为多个部分，参照word/phrases in the kb + existing word segmentation tools especially for shorter words
- clarify question structure
- ![1586398092065](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019/pictures/1586398092065.png)



## Multi-Module System for Open Domain Chinese Question Answering over Knowledge Base

### 结果

- F1：70.45

### 系统

#### Topic Entity Linking

- entity mention dictionary
- paddle-paddle ner module："LOC", "ORG", "PER", "TIME", none, person name, place name, institution name, work name
- financial field wordlist
- stopwords for relation, only valid when there are no words detected in the ner module
- scoring module:
  - length of entity, the longer the higher
  - out-degree of entity in the kb
  - **the distance between entity and interrogative words, an entity gets higher score if more close to the ['who', 'what', 'where, 'how', 'how much, 'how many']**
  - char overlapping between entity and question
  - word overlap between entity-mention and question
  - NER label of entity
  - similarity between entity-mention and question: finetuned bert
  - retrieve top2 entities

#### Relation Recognition

- score relation similarity: bert-base + bilstm + fc to get semantic embeddings, cosine comparation
- score object similarity: same as score relation similarity, second indicator of relation recognition
- score char overlap: added primarily to prevent the model from relying too heavily on the bert similarity model
- weighted addition of the 3 scores above

#### Answer Selection

- simple-complex question classifier - bert classifier
- sparql generation: 5 structures
- ![1586389192332](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019/pictures/1586389192332.png)

## DUTIR 中文开放域知识库问答评测报告

### 结果

- F1：67.6

### 模型

#### 辅助词典构建

- 实体链接词典：由主办方提供
- 分词词典：实体链接词典中的所有实体提及，知识库中所有实体的主干成分
- 词频词典：计算实体提及和属性值提及的词频特征，利用搜狗开源中文词频词典构建
- 倒排索引：识别属性值的模糊匹配

#### 实体提及和属性值提及识别

- bert将训练集中标注实体还原为实体提及：“大连理工的校歌|是|什么？” -> "大连理工|的|校歌|是什么"
- 属性值提及识别
  - 书名，称号，数字，正则
  - 时间属性，正则
  - 模糊匹配属性：得到问题中每个字对应的所有属性值，统计每个属性值的次数，选top3加入候选属性值的提及

#### 实体链接及筛选

- （1）实体提及的长度：该实体对应的实体提及的字数；
  （2）实体提及的词频：该实体对应的实体提及的词频；
  （3）实体提及的位置：该实体对应的实体提及距离句首的距离；
  （4）实体两跳内关系和问题重叠词的数量；
  （5）实体两跳内关系和问题重叠字的数量；
- logistic回归进行训练 打分 预测

### 候选查询路径生成及文本匹配

- 对每个实体抽取单跳关系和两跳关系作为候选的查询语句
- bert [cls] q1 [seg] 查询路径还原的人工问题 [seg] 进行打分

#### 桥接及答案选择

- 有一部分包含两个及以上的主语实体，例如“北京大学除了哪些哲学家”
- 对匹配的单跳候选路径到知识库进行检索，验证其是否能和其他候选实体组成多实体情况的查询路径 {ent1, rel1, ANSWER, rel2, ent2}



## AliMe KBQA: Question Answering over Structured Knowledge for E-commerce Customer Service

### Knowledge Representation

![1586439182270](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019/pictures/1586439182270.png)

![1586439376518](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019/pictures/1586439376518.png)

CVT（Compound Value Type）is obtained from Freebase.

- 营销工具-店铺宝类似于我们的Subgenre-Instance
- 这个property也可以是多级的，目前我们的模型可能不需要
- 收费标准中的CVT有点像我们的限制类，就是对某一个属性再加上其中一个限制，这里就是business和merchant rating这种
- the structured representation largely reduces the number of knowledge items, enabling convenient knowledge management and better model matching performance.
- 新加入schema也比较快

### KBQA Approach

![1586442328957](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2019/pictures/1586442328957.png)

- 通过先生成基本查询图的形式(e, p, v)，然后entity-linking通过规则匹配方式做，然后通过tailored CNN映射到property上。
- 在 v 是CVT node 的时候，通过rule-based 和 similarity-based 字符串匹配来将它链接到正确的v上（通过打分排序）