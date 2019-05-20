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


if __name__ == '__main__':
    # zero的数据上限
    zero_threshold = 3000
    data_analysis('../raw_data/open_data/sent_relation_train.txt')
    multi_flag = True if sys.argv[1] == 'multi' else False
    # 多任务方法
    if multi_flag:
        # train file
        sent_file = '../raw_data/open_data/sent_train.txt'
        label_file = '../raw_data/open_data/sent_relation_train.txt'
        out_file = '../data/sent_train_multi.txt'
        generate_json_data_multi_remove(sent_file, label_file, out_file, flag='train')

        # dev file
        sent_file = '../raw_data/open_data/sent_dev.txt'
        label_file = '../raw_data/open_data/sent_relation_dev.txt'
        out_file = '../data/sent_dev_multi.txt'
        generate_json_data_multi_remove(sent_file, label_file, out_file, flag='dev')

        # test file
        sent_file = '../raw_data/open_data/sent_test.txt'
        label_file = '../raw_data/open_data/sent_relation_test.txt'
        out_file = '../data/sent_test_multi.txt'
        generate_json_data_multi_remove(sent_file, label_file, out_file, flag='test')
    # 普通方法
    else:
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

