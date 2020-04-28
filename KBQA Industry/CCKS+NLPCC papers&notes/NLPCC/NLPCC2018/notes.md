# 2018 NLPCC

## 基于 LSTM 的大规模知识库自动问答

### 实验

测试集F1：81.06

### 系统

#### 知识库预处理

+ 去除属性中空白字符
+ 去除属性中所有非中文、数字和英文字母的字符
+ 将实体和属性中的所有大写外文字转为小写

#### 命名实体识别

+ 别名词典
  + 带有类似别名的属性（以“名”，“称”，“名称”结尾）
  + 去除括号，书名号
+ LSTM语言模型特征
  + 根据别名词典找出问句中实体，替换成\_NER\_
  + 单向LSTM，\_NER_部分初始为0向量，不断迭代训练
+ 词表面特征
  + 命名实体长度
  + 命名实体IDF
+ 最后将上述三个进行Logistic回归

#### 属性映射

问句-属性 对进行编码

+ 结合静态注意力机制的双向LSTM

  ![1587405868485](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/NLPCC/NLPCC2018/pictures/1587405868485.png)

+ 基于单词语义相似度的注意力机制

  ![1587406334977](https://raw.githubusercontent.com/BDBC-KG-NLP/KBQA-Survey/master/KBQA%20Industry/CCKS%2BNLPCC%20papers%26notes/NLPCC/NLPCC2018/pictures/1587406334977.png)

  + S_p和S_q分别表示问句的词向量矩阵和属性的词向量矩阵
  + 注意力矩阵A中第i行第j列表示问句中第i个单词和属性第j个短语的语义相似度
  + 分别计算同维度的R_p和R_q并拼接到原S_p S_q上，送入同上的单层LSTM中

+ 送入logistic回归进行拟合

#### 答案选择

+ 对NER_SCORE和PROP_SCORE进行加权，得到Score=aNer_SCORE + (1-a)PROP_SCORE
+ 一定程度上可以通过属性进行消歧

