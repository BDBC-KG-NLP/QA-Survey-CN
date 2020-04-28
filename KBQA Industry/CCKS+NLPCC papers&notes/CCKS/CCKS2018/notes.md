# CCKS 2018 论文笔记

## A QA search algorithm based on the fusion integration of text similarity and graph computation
### 结果

+ F1：NaN (1st)

### 系统

![1587283455669](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2018/pictures/1587283455669.png)

#### Text Similarity Search Model

+ 问题subject实体和/或object实体包含更多信息。

  + Use object entity index to ?nd whole sentence entity
  + Discover entity using mention index
  +  Identify the predicate entity or type entity that may appear before main entity

+ 1-degree problem：一个subject/object + predicate

+ 2-degree problem：可以找到第二个triple，共六种匹配方式(相似度)

  + 1）S-P-<>-P-?
    The main entity starts as a subject and queries its indirect attribute values.
    For example：《蒙娜丽莎》的作者皈依的是什么宗教？赵明诚的配偶有哪些主要作品？

  + 2）S-<>-O-P-?
    Similar to 1), but the subject entity is not directly related to the attribute, but its object entity.
    for example：澳大利亚的悉尼有什么著名景点？

  + 3）S-P-?-P-<>
    Similar to 1), but its problem is another subject of the object entity of main entity.
    for example：周恩来的妻子曾当过什么主席？

  + 4）O-P-S-P-?
    The main entity starts as an object entity and queries its indirect attribute values.
    For example：小说《哈利波特》的作者是谁？郦道元的《水经注》编撰于哪个朝代？

  + 5）O-<>-?-P-O1
    The main entity starts as an object entity, and with the aid of other entities in the question, queries its subject entity.
    For example：北京大学出了哪些科学家？理查德·格里格和菲利普·津巴多合著的书是什么？

  + 6）O-P-?-<>-O1
    Similar to 5), but predicate information is the parent attribute of main entity.

    For example：有哪些位于湖北的公司？

#### Graph computation model

![1587299540999](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2018/pictures/1587299540999.png)

+ Pre-solution space：找实体两跳(正反)内的子图
+ Structural Feature
  + jaccard distance
  + common characters ratio
  + edit distance
  + 疑问词和目标词距离、
+ Node consolidation：两个实体中出现公用部分，进行融合。e.g.  张柏芝,谢霆锋共同主演的电影
+ Pruning：
  + use heuristic rules for pruning
  + 使用决策树，将“ one or several indicators“的平均值记下，控制之前选过的节点中用同种方法计算，大于该平均值的个数不超过80
+ Similarity Features
  + 用CNN编码路径和问题的相似度

## A Joint Model of Entity Linking and Predicate Recognition for Knowledge Base Question Answering

### 结果

- F1：57.67% (2nd)

### 系统

![1587280336078](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2018/pictures/1587280336078.png)

### Entity Mention Recognition

+ Segmentation Dic是所有KB中实体以及在CCKS给的Mention Dic中的提及
+ 基于规则：
  + 实体提及长度
  + 实体提及的TF值
  + 实体距离疑问词的距离
  + 进行打分

### Entity Linking

+ 基于规则和语义，拿出实体两跳内的子图
  + 问题和三元组路径的词语覆盖率
  + 问题和三元组路径word embedding相似度
  + 问题和三元组路径的字符覆盖率

### Overall Entity Score

weighted linear score：$Score_{topicentity}=w_1*F_1+w_2*F_2+w_3*F_3+w_4*F_4+w_5*F_5+w_6*F_6$

### Predicate Recognition

+ extract 4 features of triple path:
  + 问题和谓词的词语覆盖率
  + 问题和谓词的word embedding相似度
  + 问题和谓词的字符覆盖率
  + 问题和谓词的char embedding相似度
+ rank top 10

### Semantic Matching

+ word embedding + 10 features above

![1587281670040](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/CCKS/CCKS2018/pictures/1587281670040.png)

+ 网络：BiMPM+Fea
  + Word Representation Layer: 将每个word映射到100维
  + Context Representation Layer：按照BiLSTM来encode每一跳
  + Matching Layer：BiLSTM+similarity
  + Aggregation Layer：aggregate question and triple path to fixed length
  + Feature Aggregation Layer：Aggregation Layer 拼接 10 features

### Answer Selection

+ 分为1hop和2hop，其中2hop区分答案在中心节点还是叶子节点