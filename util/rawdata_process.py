# -*- coding:utf-8 -*-
import codecs
import json
import sys


# 将多个文件的训练数据合并转化成json格式便于处理
def generate_json_data(sent_file, label_file, out_file, flag):
    if flag != 'test':
        with codecs.open(sent_file, encoding='utf-8') as sent, codecs.open(label_file, encoding='utf-8') as label, codecs.open(out_file, 'w', encoding='utf-8') as fw:
            for line_sent in sent:
                line_label = label.readline()
                line_sent = line_sent.strip('\r\n').split('\t')
                line_label = line_label.strip('\r\n').split('\t')
                json_dict = {'id': line_sent[0], 'per1': line_sent[1], 'per2': line_sent[2], 'sent': line_sent[3],
                             'label': line_label[1]}
                fj = json.dumps(json_dict, ensure_ascii=False)
                fw.write(fj + '\n')
        print("json data generated!!!")
    else:
        with codecs.open(sent_file, encoding='utf-8') as sent, codecs.open(out_file, 'w', encoding='utf-8') as fw:
            for line_sent in sent:
                line_sent = line_sent.strip('\r\n').split('\t')
                json_dict = {'id': line_sent[0], 'per1': line_sent[1], 'per2': line_sent[2], 'sent': line_sent[3]}
                fj = json.dumps(json_dict, ensure_ascii=False)
                fw.write(fj + '\n')
        print("json data generated!!!")
    return


# 将多个文件的训练数据合并转化成json格式, 这里采用多任务方法，将标签处理为二级，label, label2;
# label表示原来的label, label2中包含两个0或者1表示0或者其他类
def generate_json_data_multi(sent_file, label_file, out_file, flag):
    if flag != 'test':
        with codecs.open(sent_file, encoding='utf-8') as sent, codecs.open(label_file, encoding='utf-8') as label, codecs.open(out_file, 'w', encoding='utf-8') as fw:
            for line_sent in sent:
                line_label = label.readline()
                line_sent = line_sent.strip('\r\n').split('\t')
                line_label = line_label.strip('\r\n').split('\t')
                label2 = '0'
                if line_label[1] != '0':
                    label2 = '1'
                json_dict = {'id': line_sent[0], 'per1': line_sent[1], 'per2': line_sent[2], 'sent': line_sent[3],
                             'label': line_label[1], 'label2': label2}
                fj = json.dumps(json_dict, ensure_ascii=False)
                fw.write(fj + '\n')
        print("json data generated!!!")
    else:
        with codecs.open(sent_file, encoding='utf-8') as sent, codecs.open(out_file, 'w', encoding='utf-8') as fw:
            for line_sent in sent:
                line_sent = line_sent.strip('\r\n').split('\t')
                json_dict = {'id': line_sent[0], 'per1': line_sent[1], 'per2': line_sent[2], 'sent': line_sent[3]}
                fj = json.dumps(json_dict, ensure_ascii=False)
                fw.write(fj + '\n')
        print("json data generated!!!")
    return


# 计算数据分布
def data_analysis(label_file):
    label_dict = dict()
    with codecs.open(label_file, encoding='utf-8') as labels:
        for label in labels:
            line_label = label.strip('\r\n').split('\t')[1]
            if line_label not in label_dict:
                label_dict[line_label] = 1
            else:
                label_dict[line_label] = label_dict[line_label] + 1
    for i, v in label_dict.items():
        print(i, v)
    return


# 内部原始词向量语料
def word_corpus_generate(train_file, dev_file, test_file, out_file):
    with codecs.open(train_file, encoding='utf-8') as sent, codecs.open(out_file, 'w', encoding='utf-8') as fw:
        for line_sent in sent:
            line_sent = line_sent.strip('\r\n').split('\t')
            fw.write(line_sent[3] + '\n')
    with codecs.open(dev_file, encoding='utf-8') as sent, codecs.open(out_file, 'a', encoding='utf-8') as fw:
        for line_sent in sent:
            line_sent = line_sent.strip('\r\n').split('\t')
            fw.write(line_sent[3] + '\n')
    with codecs.open(test_file, encoding='utf-8') as sent, codecs.open(out_file, 'a', encoding='utf-8') as fw:
        for line_sent in sent:
            line_sent = line_sent.strip('\r\n').split('\t')
            fw.write(line_sent[3] + '\n')
    return


# 将多个文件的训练数据合并转化成json格式, 这里采用多任务方法，将标签处理为二级，label, label2;
# label表示原来的label, label2中包含两个0或者1表示0或者其他类
# 去掉一部分0数据
def generate_json_data_multi_remove(sent_file, label_file, out_file, flag):
    if flag != 'test':
        with codecs.open(sent_file, encoding='utf-8') as sent, codecs.open(label_file, encoding='utf-8') as label, codecs.open(out_file, 'w', encoding='utf-8') as fw:
            zero_count = 0
            for line_sent in sent:
                line_label = label.readline()
                line_sent = line_sent.strip('\r\n').split('\t')
                line_label = line_label.strip('\r\n').split('\t')
                label2 = '0'
                if flag == 'train' and line_label[1] == '0' and zero_count >= zero_threshold:
                    continue
                else:
                    zero_count += 1
                if line_label[1] != '0':
                    label2 = '1'
                json_dict = {'id': line_sent[0], 'per1': line_sent[1], 'per2': line_sent[2], 'sent': line_sent[3],
                             'label': line_label[1], 'label2': label2}
                fj = json.dumps(json_dict, ensure_ascii=False)
                fw.write(fj + '\n')
        print("json data generated!!!")
    else:
        with codecs.open(sent_file, encoding='utf-8') as sent, codecs.open(out_file, 'w', encoding='utf-8') as fw:
            for line_sent in sent:
                line_sent = line_sent.strip('\r\n').split('\t')
                json_dict = {'id': line_sent[0], 'per1': line_sent[1], 'per2': line_sent[2], 'sent': line_sent[3]}
                fj = json.dumps(json_dict, ensure_ascii=False)
                fw.write(fj + '\n')
        print("json data generated!!!")
    return


# 将数据划分为10份，NA类分为10份，其他类重复出现在这10份中
def divide_data_10_fold(sent_file, label_file):
    sent = codecs.open(sent_file, encoding='utf-8')
    label = codecs.open(label_file, encoding='utf-8')
    zero_count = 1
    fold = 0
    fw = codecs.open('../data/fold/sent_train' + str(fold) + '.txt', 'w', encoding='utf-8')
    other_class = []
    for line_sent in sent:
        line_label = label.readline()
        line_sent = line_sent.strip('\r\n').split('\t')
        line_label = line_label.strip('\r\n').split('\t')
        if line_label[1] == '0':
            if fold > 9:
                fw.close()
                continue
            if zero_count % zero_threshold == 0:
                fold += 1
                fw.close()
                fw = codecs.open('../data/fold/sent_train' + str(fold) + '.txt', 'w', encoding='utf-8')
            zero_count += 1
            json_dict = {'id': line_sent[0], 'per1': line_sent[1], 'per2': line_sent[2], 'sent': line_sent[3],
                         'label': line_label[1]}
            fj = json.dumps(json_dict, ensure_ascii=False)
            fw.write(fj + '\n')
        else:
            json_dict = {'id': line_sent[0], 'per1': line_sent[1], 'per2': line_sent[2], 'sent': line_sent[3],
                         'label': line_label[1]}
            fj = json.dumps(json_dict, ensure_ascii=False)
            other_class.append(fj)
    for i in range(10):
        fw = codecs.open('../data/fold/sent_train' + str(i) + '.txt', 'a', encoding='utf-8')
        for line in other_class:
            fw.write(line + '\n')
        fw.close()
    print("json data divided!!!")
    return


if __name__ == '__main__':
    # zero的数据上限
    zero_threshold = 20000
    data_analysis('../raw_data/open_data/sent_relation_train.txt')
    # 产生内部的词库
    # word_corpus_generate('../raw_data/open_data/sent_train.txt', '../raw_data/open_data/sent_dev.txt', '../raw_data/open_data/sent_test.txt', '../data/word_level/corpus_raw.txt')
    flag = 'fold'
    # 多任务方法
    if flag == 'multi':
        # train file
        sent_file = '../raw_data/open_data/sent_train.txt'
        label_file = '../raw_data/open_data/sent_relation_train.txt'
        out_file = '../data/sent_train_multi.txt'
        generate_json_data_multi(sent_file, label_file, out_file, flag='train')
        # 去掉部分数据效果不好
        # generate_json_data_multi_remove(sent_file, label_file, out_file, flag='train')

        # dev file
        sent_file = '../raw_data/open_data/sent_dev.txt'
        label_file = '../raw_data/open_data/sent_relation_dev.txt'
        out_file = '../data/sent_dev_multi.txt'
        generate_json_data_multi(sent_file, label_file, out_file, flag='dev')

        # test file
        sent_file = '../raw_data/open_data/sent_test.txt'
        label_file = '../raw_data/open_data/sent_relation_test.txt'
        out_file = '../data/sent_test_multi.txt'
        generate_json_data_multi(sent_file, label_file, out_file, flag='test')
    # 普通方法
    elif flag == 'single':
        # train file
        sent_file = '../raw_data/open_data/sent_train.txt'
        label_file = '../raw_data/open_data/sent_relation_train.txt'
        out_file = '../data/sent_train.txt'
        generate_json_data(sent_file, label_file, out_file, flag='train')

        # dev file
        sent_file = '../raw_data/open_data/sent_dev.txt'
        label_file = '../raw_data/open_data/sent_relation_dev.txt'
        out_file = '../data/sent_dev.txt'
        generate_json_data(sent_file, label_file, out_file, flag='dev')

        # test file
        sent_file = '../raw_data/open_data/sent_test.txt'
        label_file = '../raw_data/open_data/sent_relation_test.txt'
        out_file = '../data/sent_test.txt'
        generate_json_data(sent_file, label_file, out_file, flag='test')
    elif flag == 'fold':
        sent_file = '../raw_data/open_data/sent_train.txt'
        label_file = '../raw_data/open_data/sent_relation_train.txt'
        divide_data_10_fold(sent_file, label_file)

