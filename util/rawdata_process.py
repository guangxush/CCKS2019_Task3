# -*- coding:utf-8 -*-
import codecs
import json


def generate_json_data(sent_file, label_file, out_file):
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
    return


if __name__ == '__main__':
    # train file
    sent_file = '../raw_data/open_data/sent_train.txt'
    label_file = '../raw_data/open_data/sent_relation_train.txt'
    out_file = '../data/sent_train.txt'
    generate_json_data(sent_file, label_file, out_file)

    # dev file
    sent_file = '../raw_data/open_data/sent_dev.txt'
    label_file = '../raw_data/open_data/sent_relation_dev.txt'
    out_file = '../data/sent_dev.txt'
    generate_json_data(sent_file, label_file, out_file)
