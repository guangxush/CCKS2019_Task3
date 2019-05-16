# -*- encoding:utf-8 -*-
import os
from util.data_process import load_data
from models import models as Models
from config import Config
import numpy as np


def get_data(train_file=None, valid_file=None, test_file=None, level='word'):
    if level == 'char' or level == 'word' or level == 'test':
        x_train, y_train, vocabulary = load_data(train_file, 'word')
        x_valid, y_valid, vocabulary = load_data(valid_file, 'word')
    # if level == 'test':
        ids, x_test, vocabulary = load_data(test_file, 'test')
    return x_train, y_train, x_valid, y_valid, x_test, vocabulary, ids


def siamese_cnn(x_train_a, x_train_b, y_train, x_valid_a, x_valid_b, y_valid, x_test_a, x_test_b, y_test, level,
        fasttext=False, overwrite=False, distance=False, manhattan=False):
    config = Config()
    config.level = level
    if level == 'word':
        config.max_len = config.max_len_word
        config.vocab_len = config.vocab_len_word
    else:
        config.max_len = config.max_len_char
        config.vocab_len = config.vocab_len_char
    config.exp_name = 'siamese_cnn_' +level
    if distance:
        config.exp_name += '_distance'
    if manhattan:
        config.exp_name += '_manhattan'
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if fasttext:
        config.embedding_file += 'fasttext'
        config.exp_name += '_fasttext'
    else:
        config.embedding_file += 'embeddings'
    siamese_model = Models(config)
    print('Create the siamese_cnn model...')
    siamese_model.siamese_cnn(distance=distance, manhattan=manhattan)
    if overwrite or not os.path.exists(os.path.join(config.checkpoint_dir, '%s.hdf5' % config.exp_name)):
        print('Start training the siamese_cnn model...')
        siamese_model.fit(x_train_a, x_train_b, y_train, x_valid_a, x_valid_b, y_valid, distance=distance)
    siamese_model.load_weight()
    print('Start evaluate the siamese_cnn model...')
    y_valid_pred = siamese_model.predict(x_valid_a, x_valid_b)
    y_test_pred = siamese_model.predict(x_test_a, x_test_b)
    siamese_model.evaluate(y_valid_pred, y_valid, distance=distance)
    siamese_model.evaluate(y_test_pred, y_test, distance=distance)


def siamese_att_cnn(x_train_a, x_train_b, y_train, x_valid_a, x_valid_b, y_valid, x_test_a, x_test_b, y_test, level,
        fasttext=False, overwrite=False, distance=False, manhattan=False):
    config = Config()
    config.level = level
    if level == 'word':
        config.max_len = config.max_len_word
        config.vocab_len = config.vocab_len_word
    else:
        config.max_len = config.max_len_char
        config.vocab_len = config.vocab_len_char
    config.exp_name = 'siamses_att_cnn_' +level
    if distance:
        config.exp_name += '_distance'
    if manhattan:
        config.exp_name += '_manhattan'
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if fasttext:
        config.embedding_file += 'fasttext'
        config.exp_name += '_fasttext'
    else:
        config.embedding_file += 'embeddings'
    siamese_model = Models(config)
    print('Create the siamese_att_cnn model...')
    siamese_model.siamese_att_cnn(distance=distance, manhattan=manhattan)
    if overwrite or not os.path.exists(os.path.join(config.checkpoint_dir, '%s.hdf5' % config.exp_name)):
        print('Start training the siamese_att_cnn model...')
        siamese_model.fit(x_train_a, x_train_b, y_train, x_valid_a, x_valid_b, y_valid, distance=distance)
    siamese_model.load_weight()
    print('Start evaluate the siamese_att_cnn model...')
    y_valid_pred = siamese_model.predict(x_valid_a, x_valid_b)
    y_test_pred = siamese_model.predict(x_test_a, x_test_b)
    siamese_model.evaluate(y_valid_pred, y_valid, distance=distance)
    siamese_model.evaluate(y_test_pred, y_test, distance=distance)


def cnn_base(x_train, y_train, x_valid, y_valid, x_test, level, fasttext=False, overwrite=False):
    config = Config()
    config.level = level
    if level == 'word':
        config.max_len = config.max_len_word
        config.vocab_len = config.vocab_len_word
    else:
        config.max_len = config.max_len_char
        config.vocab_len = config.vocab_len_char
    config.exp_name = 'cnn_base_' +level

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if fasttext:
        config.embedding_file += 'fasttext'
        config.exp_name += '_fasttext'
    else:
        config.embedding_file += 'embeddings'
    cnn_model = Models.Models(config)
    print('Create the cnn model...')
    cnn_model.cnn_base()
    if overwrite or not os.path.exists(os.path.join(config.checkpoint_dir, '%s.hdf5' % config.exp_name)):
        print('Start training the cnn model...')
        cnn_model.fit(x_train, y_train, x_valid, y_valid)
    cnn_model.load_weight()
    print('Start evaluate the cnn model...')
    y_valid_pred = cnn_model.predict(x_valid)
    y_test_pred = cnn_model.predict(x_test)
    cnn_model.evaluate(y_valid_pred, y_valid)
    print('Start generate the cnn model...')

    return y_test_pred


def generate_result(ids, y_test_pred):
    config = Config()
    fw = open(config.result_file, 'w')
    line = 0
    for id in ids:
        y_test = np.argmax(y_test_pred[line])
        line += 1
        fw.write(str(id) + '\t' + str(y_test) + '\n')
    fw.close()
    return


if __name__ == '__main__':
    level = 'word'
    fasttext = False
    overwrite = False
    print('Load %s_level data...' % level)
    x_train, y_train, x_valid, y_valid, x_test, vocab, ids = \
        get_data(train_file='./data/sent_train.txt', valid_file='./data/sent_dev.txt',
                 test_file='./data/sent_test.txt', level=level)
    y_test_pred = cnn_base(x_train, y_train, x_valid, y_valid, x_test, level, fasttext=fasttext, overwrite=overwrite)
    generate_result(ids, y_test_pred)

