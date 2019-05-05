# -*- encoding:utf-8 -*-
import os
from util.data_process import load_data
from models import models as Models
from config import Config


def get_data(train_file=None, valid_file=None, test_file=None, level='word'):
    if level == 'char' or level == 'word':
        x_train_a, x_train_b, y_train, vocabulary = load_data(train_file, level)
        x_valid_a, x_valid_b, y_valid, vocabulary = load_data(valid_file, level)
        x_test_a, x_test_b, y_test, vocabulary = load_data(test_file, level)
        return x_train_a, x_train_b, y_train, x_valid_a, x_valid_b, y_valid, x_test_a, x_test_b, y_test, vocabulary


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


if __name__ == '__main__':
    level = 'word'
    fasttext = False
    overwrite = False
    print('Load %s_level data...' % level)
    x_train_a, x_train_b, y_train, x_valid_a, x_valid_b, y_valid, x_test_a, x_test_b, y_test, vocab = \
        get_data(train_file='data/xxx_train.tsv', valid_file='data/xxx_dev.tsv',
                 test_file='data/xxx_test.tsv', level=level)

    siamese_cnn(x_train_a, x_train_b, y_train, x_valid_a, x_valid_b, y_valid, x_test_a, x_test_b, y_test, level,
                    fasttext=fasttext, overwrite=overwrite)
    siamese_att_cnn(x_train_a, x_train_b, y_train, x_valid_a, x_valid_b, y_valid, x_test_a, x_test_b, y_test, level,
                    fasttext=fasttext, overwrite=overwrite)
