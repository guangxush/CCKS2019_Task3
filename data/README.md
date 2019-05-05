#### save the processed data

CCKS2019-IPRE任务数据集介绍

- 训练集
  1) sent_train.txt
     训练集中的实体对、句子的组合（以下简称为一个实例）及其ID

  2) sent_relation_train.txt
	 训练集中每个实例及其对应的关系

  3) bag_relation_train.txt
	 训练集中的包、包中的实例及包的关系标签

- 验证集
  1) sent_dev.txt
     验证集中的实体对、句子的组合及其ID

  2) sent_relation_dev.txt
	 验证集中每个实例及其对应的关系

  3) bag_relation_dev.txt
	 验证集中的包、包中的实例及包的关系标签

- 测试集
  1) sent_test.txt
     测试集中的实体对、句子的组合及其ID

  2) sent_relation_test.txt
	 Sent-Track测试集中每个实例

  3) bag_relation_test.txt
	 Bag-Track测试集中的包及包中的实例

- 关系表
  1) relation2id.txt
     人物关系及其ID

- 文本语料
  1) text.txt
     用于训练词向量和语言模型的大规模无标注语料