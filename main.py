# -*- encoding:utf-8 -*-
import os
from util.data_process import load_data, load_data_multi_dis
from models import models as Models
from config import Config
import numpy as np
import time
import sys


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


# 多任务获取数据
def get_data_multi(train_file=None, valid_file=None, test_file=None, flag='train'):
    if flag == 'train':
        x_train, x_train_dis1, x_train_dis2, y_train, y_train2, vocabulary = load_data_multi_dis(train_file, 'word')
        x_valid, x_valid_dis1, x_valid_dis2, y_valid, y_valid2, vocabulary = load_data_multi_dis(valid_file, 'word')
        ids, x_test, x_test_dis1, x_test_dis2, vocabulary = load_data_multi_dis(test_file, 'test')
        return x_train, x_train_dis1, x_train_dis2, y_train, y_train2, x_valid, x_valid_dis1, x_valid_dis2, y_valid, y_valid2, x_test, x_test_dis1, x_test_dis2, vocabulary, ids
    elif flag == 'test':
        ids, x_test, disinfos1, disinfos2, vocabulary = load_data_multi_dis(test_file, 'test')
        return x_test, disinfos1, disinfos2, vocabulary, ids


# 创建多任务模型
def model_select_multi(model_name, x_train, x_train_dis1, x_train_dis2, y_train, y_train2, x_valid, x_valid_dis1,
                       x_valid_dis2, y_valid, y_valid2, x_test, x_test_dis1, x_test_dis2, level, overwrite=False):
    config = Config()
    config.level = level
    if level == 'word':
        # 固定最大长度，多余的截取掉，不足的用0填充
        config.max_len = config.max_len_word
        config.vocab_len = config.vocab_len_word
    else:
        config.max_len = config.max_len_char
        config.vocab_len = config.vocab_len_char
    config.exp_name = 'cnn_base_multi'

    # 训练的模型保存成文件的形式
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    # 获取词向量文件
    config.embedding_file += 'embeddings'
    # 载入配置文件
    model = Models.Models(config)

    # 模型训练
    print('Create the multi task ' + model_name + ' model...')
    if model_name == 'cnn_base_multi':
        model.cnn_multi_base()
    elif model_name == 'bilstm_base_multi':
        model.bilstm_multi_base()
    else:
        return
    if overwrite or not os.path.exists(os.path.join(config.checkpoint_dir, '%s.hdf5' % model_name)):
        print('Start training the multi task ' + model_name + ' model...')
        model.fit_multi_dis(x_train, x_train_dis1, x_train_dis2, y_train, y_train2, x_valid, x_valid_dis1, x_valid_dis2,
                            y_valid, y_valid2)
    model.load_weight()
    print('Start evaluate the multi task ' + model_name + ' model...')
    y_valid_pred = model.predict_multi(x_valid, x_valid_dis1, x_valid_dis2)
    y_test_pred = model.predict_multi(x_test, x_test_dis1, x_test_dis2)
    model.evaluate(model_name, y_valid_pred, y_valid)
    print('Start generate the multi task ' + model_name + ' model...')

    return y_test_pred


# 创建单一任务模型
def model_select(model_name, x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1, x_valid_dis2, y_valid,
                 x_test, x_test_dis1, x_test_dis2, level, overwrite=False):
    config = Config()
    config.level = level
    if level == 'word':
        # 固定最大长度，多余的截取掉，不足的用0填充
        config.max_len = config.max_len_word
        config.vocab_len = config.vocab_len_word
    else:
        config.max_len = config.max_len_char
        config.vocab_len = config.vocab_len_char
    config.exp_name = model_name

    # 训练的模型保存成文件的形式
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    # 获取词向量文件
    config.embedding_file += 'embeddings'
    # 载入配置文件
    model = Models.Models(config)

    # 模型训练
    print('Create the ' + model_name + ' model...')
    if model_name == 'cnn_base':
        model.cnn_base()
    elif model_name == 'bilstm_base':
        model.bilstm_base()
    else:
        return

    if overwrite or not os.path.exists(os.path.join(config.checkpoint_dir, '%s.hdf5' % model_name)):
        print('Start training the ' + model_name + ' model...')
        model.fit(x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1, x_valid_dis2, y_valid)
    model.load_weight()
    print('Start evaluate the ' + model_name + ' model...')
    y_valid_pred = model.predict(x_valid, x_valid_dis1, x_valid_dis2)
    y_test_pred = model.predict(x_test, x_test_dis1, x_test_dis2)
    model.evaluate(model_name, y_valid_pred, y_valid)
    print('Start generate the ' + model_name + ' model...')

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

    multi_flag = True if sys.argv[1] == 'multi' else False
    if multi_flag:
        x_train, x_train_dis1, x_train_dis2, y_train, y_train2, x_valid, x_valid_dis1, x_valid_dis2, y_valid, y_valid2, x_test, x_test_dis1, x_test_dis2, vocabulary, ids = \
            get_data_multi(train_file='./data/sent_train_multi.txt', valid_file='./data/sent_dev_multi.txt',
                           test_file='./data/sent_test_multi.txt', flag='train')

        # cnn base model
        y_test_pred = model_select_multi('cnn_base_multi', x_train, x_train_dis1, x_train_dis2, y_train, y_train2,
                                         x_valid, x_valid_dis1, x_valid_dis2, y_valid, y_valid2, x_test, x_test_dis1,
                                         x_test_dis2, level, overwrite=overwrite)

        # y_test_pred = model_select_multi('bilstm_base_multi', x_train, x_train_dis1, x_train_dis2, y_train, y_train2,
        #                                  x_valid, x_valid_dis1, x_valid_dis2, y_valid, y_valid2, x_test, x_test_dis1,
        #                                  x_test_dis2, level, overwrite=overwrite)

        generate_result(ids, y_test_pred)
    else:
        x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1, x_valid_dis2, y_valid, x_test, x_test_dis1, x_test_dis2, vocabulary, ids = \
            get_data(train_file='./data/sent_train_multi.txt', valid_file='./data/sent_dev_multi.txt',
                     test_file='./data/sent_test_multi.txt', flag='train')

        # cnn base model
        y_test_pred = model_select('cnn_base', x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1,
                                   x_valid_dis2, y_valid,
                                   x_test, x_test_dis1, x_test_dis2, level, overwrite=overwrite)

        # bilstm model
        # y_test_pred = model_select('bilstm_base', x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1,
        #                            x_valid_dis2, y_valid, x_test, x_test_dis1, x_test_dis2, level, overwrite=overwrite)

        generate_result(ids, y_test_pred)
