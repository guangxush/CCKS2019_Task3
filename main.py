# -*- encoding:utf-8 -*-
import os
from util.data_process import load_data
from models import models as Models
from config import Config
import numpy as np
import time


# 获取数据
def get_data(train_file=None, valid_file=None, test_file=None, flag='train'):
    if flag == 'train':
        x_train, x_train_dis1, x_train_dis2, y_train, vocabulary = load_data(train_file, 'word')
        x_valid, x_valid_dis1, x_valid_dis2, y_valid, vocabulary = load_data(valid_file, 'word')
        ids, x_test, x_test_dis1, x_test_dis2, vocabulary = load_data(test_file, 'test')
        return x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1, x_valid_dis2, y_valid, x_test, x_test_dis1, x_test_dis2, vocabulary, ids
    elif flag == 'test':
        ids, x_test, disinfos1, disinfos2, vocabulary = load_data(test_file, 'test')
        return x_test, disinfos1, disinfos2, vocabulary, ids


# 创建模型
def cnn_base(x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1, x_valid_dis2, y_valid, x_test, x_test_dis1, x_test_dis2, level, overwrite=False):
    config = Config()
    config.level = level
    model_name = 'cnn_base'
    if level == 'word':
        # 固定最大长度，多余的截取掉，不足的用0填充
        config.max_len = config.max_len_word
        config.vocab_len = config.vocab_len_word
    else:
        config.max_len = config.max_len_char
        config.vocab_len = config.vocab_len_char
    config.exp_name = 'cnn_base'

    # 训练的模型保存成文件的形式
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    # 获取词向量文件
    config.embedding_file += 'embeddings'
    # 载入配置文件
    cnn_model = Models.Models(config)

    # 模型训练
    print('Create the cnn model...')
    cnn_model.cnn_base()
    if overwrite or not os.path.exists(os.path.join(config.checkpoint_dir, '%s.hdf5' % model_name)):
        print('Start training the cnn model...')
        cnn_model.fit(x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1, x_valid_dis2, y_valid)
    cnn_model.load_weight()
    print('Start evaluate the cnn model...')
    y_valid_pred = cnn_model.predict(x_valid, x_valid_dis1, x_valid_dis2)
    y_test_pred = cnn_model.predict(x_test, x_test_dis1, x_test_dis2)
    cnn_model.evaluate(model_name, y_valid_pred, y_valid)
    print('Start generate the cnn model...')

    return y_test_pred


# 生成预测结果
def generate_result(ids, y_test_pred):
    config = Config()
    fw = open(config.result_file + '.txt', 'w')
    t = str(int(time.time()))
    fw_time = open(config.result_file + '_' + t + '.txt', 'w')
    line = 0
    for id in ids:
        y_test = np.argmax(y_test_pred[line])
        line += 1
        fw.write(str(id) + '\t' + str(y_test) + '\n')
        fw_time.write(str(id) + '\t' + str(y_test) + '\n')
    fw.close()
    fw_time.close()
    return


if __name__ == '__main__':

    level = 'word'
    fasttext = False
    overwrite = False
    print('Load %s_level data...' % level)
    x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1, x_valid_dis2, y_valid, x_test, x_test_dis1, x_test_dis2, vocabulary, ids = \
        get_data(train_file='./data/sent_train_multi.txt', valid_file='./data/sent_dev_multi.txt',
                 test_file='./data/sent_test_multi.txt', flag='train')
    y_test_pred = cnn_base(x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1, x_valid_dis2, y_valid, x_test, x_test_dis1, x_test_dis2, level, overwrite=overwrite)
    generate_result(ids, y_test_pred)
