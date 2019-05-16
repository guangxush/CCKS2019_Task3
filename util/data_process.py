# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split

import os
import pickle
import numpy as np
from gensim.models import word2vec
import jieba
import codecs
import random
import logging
import json
from tqdm import tqdm
import nltk

random.seed(42)

stopwords = [u'', u' ', '\t', '.', '=']
gold_label = {'entails': 1, 'neutral': 0}


def generate_embedding(level):
    data_path = '../data/%s_level' % level
    save_model_file = '../modfile/Word2Vec.mod'
    save_model_name = 'sst_100_dim_all.embeddings'
    word_size = 100

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(os.path.join(data_path, 'corpus_all.txt'))  # 加载语料
    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
    model = word2vec.Word2Vec(sentences, min_count=1, size=word_size, window=5, workers=4)
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_name, binary=False)  # 以二进制类型保存模型以便重用

    vocab = pickle.load(open(os.path.join(data_path, 'vocabulary_all.pkl'), 'rb'))
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 2, 100), dtype='float32')
    model.save('../modfile/Word2Vec.mod')
    model.wv.save_word2vec_format('../modfile/Word2Vec.mod', binary=False)
    for w, i in vocab.items():
        if w not in d:
            continue
        emb[i, :] = weights[d[w], :]
    np.save(open('../modfile/sst_100_dim_all.embeddings', 'wb'), emb)


# def generate_fasttext_embedding(level):
#     data_path = 'data/%s_level' % level
#
#     model = train_unsupervised(input=os.path.join(data_path, 'corpus_all.txt'), model='skipgram', dim=300, epoch=10,
#                                minCount=1, wordNgrams=3)
#     # 暂时没有下载这个包，所以去掉这个方法
#
#     vocab = pickle.load(open(os.path.join(data_path, 'vocabulary_all.pkl'), 'rb'))
#     d = dict([(w, 0) for w in model.get_words()])
#     print len(d)
#     emb = np.zeros(shape=(len(vocab) + 2, 300), dtype='float32')
#     print len(vocab)
#     for w, i in vocab.items():
#         if w not in d:
#             continue
#         emb[i, :] = model.get_word_vector(w)
#     np.save(open(os.path.join(data_path, 'xxx_300_dim_all.fasttext'), 'wb'), emb)


# 训练集、验证集划分
def train_valid_split(raw_file):
    labels = list()

    with codecs.open(raw_file, encoding='utf-8') as f_raw:
        lines = f_raw.readlines()
        for line in lines:
            # _, _, label = line.strip().split('\t')
            # labels.append(label)
            line = json.loads(line)
            # input_a = line['sentence1']
            # input_b = line['sentence2']
            labels = line['gold_label']

    train_lines, valid_lines = train_test_split(lines, test_size=0.1, random_state=7, stratify=labels)

    with codecs.open('data/train.tsv', 'w', encoding='utf-8') as f_train, \
            codecs.open('data/dev.tsv', 'w', encoding='utf-8') as f_valid:
        f_train.writelines(list(train_lines))
        f_valid.writelines(list(valid_lines))


# 建立单词级别的语料库
def build_word_level_corpus_all(train_file, valid_file, test_file):
    sentences = list()

    with codecs.open(train_file, encoding='utf-8') as f_train:
        lines = f_train.readlines()
        for line in lines:
            json_data = json.loads(line)
            sentences.append(json_data['sent'])

    with codecs.open(valid_file, encoding='utf-8') as f_valid:
        lines = f_valid.readlines()
        for line in lines:
            json_data = json.loads(line)
            sentences.append(json_data['sent'])

    with codecs.open(test_file, encoding='utf-8') as f_test:
        lines = f_test.readlines()
        for line in lines:
            json_data = json.loads(line)
            sentences.append(json_data['sent'])

    target_lines = [' '.join([w for w in nltk.word_tokenize(sentence) if w not in stopwords]).lower() + '\n' for sentence in sentences]

    with codecs.open('../data/word_level/corpus_all.txt', 'w', encoding='utf-8') as f_corpus:
        f_corpus.writelines(target_lines)


def build_char_level_corpus_all(train_file, valid_file, test_file):
    sentences = list()

    with codecs.open(train_file, encoding='utf-8') as f_train:
        lines = f_train.readlines()
        for line in lines:
            input_a, input_b, label = line.strip().split('\t')
            sentences.extend([input_a, input_b])

    with codecs.open(valid_file, encoding='utf-8') as f_valid:
        lines = f_valid.readlines()
        for line in lines:
            input_a, input_b, label = line.strip().split('\t')
            sentences.extend([input_a, input_b])

    with codecs.open(test_file, encoding='utf-8') as f_test:
        lines = f_test.readlines()
        for line in lines:
            input_a, input_b, label = line.strip().split('\t')
            sentences.extend([input_a, input_b])

    target_lines = list()
    for sentence in sentences:
        target_lines.append(' '.join([char for char in sentence]) + '\n')

    with codecs.open('data/char_level/corpus_all.txt', 'w', encoding='utf-8') as f_corpus:
        f_corpus.writelines(target_lines)


# 生成词典
def build_word_level_vocabulary_all(train_file, valid_file, test_file):
    sentences = list()

    with codecs.open(train_file, encoding='utf-8') as f_train:
        lines = f_train.readlines()
        for line in lines:
            json_data = json.loads(line)
            sentences.extend(nltk.word_tokenize(json_data['sent']))

    with codecs.open(valid_file, encoding='utf-8') as f_valid:
        lines = f_valid.readlines()
        for line in lines:
            json_data = json.loads(line)
            sentences.extend(nltk.word_tokenize(json_data['sent']))

    with codecs.open(test_file, encoding='utf-8') as f_test:
        lines = f_test.readlines()
        for line in lines:
            json_data = json.loads(line)
            sentences.extend(nltk.word_tokenize(json_data['sent']))

    word_list = list(set(sentences))
    print(len(word_list))
    word_list = [word for word in word_list if word not in stopwords]

    return dict((word, idx+1) for idx, word in enumerate(word_list))


def build_word_level_vocabulary_from_out_corpus(raw_file, out_file):
    sentences = list()
    with codecs.open(raw_file, encoding='utf-8') as f_test:
        lines = f_test.readlines()
        for line in lines:
            sentence_cut = jieba.cut(line)
            sentences.append(' '.join(sentence_cut))

    with codecs.open(out_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(sentences)


def build_char_level_vocabulary_all(train_file, valid_file, test_file):
    sentences = list()

    with codecs.open(train_file, encoding='utf-8') as f_train:
        lines = f_train.readlines()
        for line in lines:
            input_a, input_b, label = line.strip().split('\t')
            sentences.extend([input_a, input_b])

    with codecs.open(valid_file, encoding='utf-8') as f_valid:
        lines = f_valid.readlines()
        for line in lines:
            input_a, input_b, label = line.strip().split('\t')
            sentences.extend([input_a, input_b])

    with codecs.open(test_file, encoding='utf-8') as f_test:
        lines = f_test.readlines()
        for line in lines:
            input_a, input_b, label = line.strip().split('\t')
            sentences.extend([input_a, input_b])

    corpus = u''.join(sentences)
    char_list = list(set([char for char in corpus]))

    return dict((char, idx+1) for idx, char in enumerate(char_list))


def load_label(raw_file):
    y = list()
    with codecs.open(raw_file, encoding='utf-8') as f_train:
        lines = f_train.readlines()
        for line in lines:
            input_a, input_b, label = line.strip().split('\t')
            y.append(float(label))
    return y


def load_raw_data(raw_file, test=False):
    with open('data/word_level/vocabulary.pkl', 'rb') as f_vocabulary:
        vocabulary = pickle.load(f_vocabulary)
    if test:
        x_a = list()
        x_b = list()
        with codecs.open(raw_file, encoding='utf-8') as f_test:
            lines = f_test.readlines()
            for line in lines:
                input_a, input_b, label = line.strip().split('\t')
                x_a.append([word for word in jieba.cut(input_a)])
                x_b.append([word for word in jieba.cut(input_b)])
        return x_a, x_b, vocabulary
    else:
        x_a = list()
        x_b = list()
        y = list()
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            for line in lines:
                input_a, input_b, label = line.strip().split('\t')
                label = gold_label[label]
                x_a.append([word for word in nltk.word_tokenize(input_a)])
                x_b.append([word for word in nltk.word_tokenize(input_b)])
                y.append(float(label))
        return x_a, x_b, y, vocabulary


def load_data(raw_file, level):

    if level == 'word':
        with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_word:', len(vocabulary))
        x_a = list()
        x_b = list()
        y = list()
        max_len = 0
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            for line in tqdm(lines):
                input_a, input_b, label = line.strip().split('\t')
                if label == '-':
                    continue
                label = gold_label[label]
                words_a = nltk.word_tokenize(input_a.lower())
                words_b = nltk.word_tokenize(input_b.lower())
                x_a.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_a if word not in stopwords])
                x_b.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_b if word not in stopwords])
                y.append(float(label))
                if len(x_a[-1]) > max_len:
                    max_len = len(x_a[-1])
                if len(x_b[-1]) > max_len:
                    max_len = len(x_b[-1])
        print('max_word_len', max_len)
        avg_len = 0
        max_len = 0
        for word, id in vocabulary.items():
            if len(word) > max_len:
                max_len = len(word)
            avg_len += len(word)
        print('char_max_len:', max_len)
        print('char_avg_len:', float(avg_len)/len(vocabulary))
        return x_a, x_b, y, vocabulary
    elif level == 'char':
        with open('data/char_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_char:', len(vocabulary))
        x_a = list()
        x_b = list()
        y = list()
        char_len = 0
        max_len = 0
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            for line in tqdm(lines):
                input_a, input_b, label = line.strip().split('\t')
                if label == '-':
                    continue
                label = gold_label[label]
                x_a.append([vocabulary.get(char, len(vocabulary) + 1) for char in input_a if char not in stopwords])
                x_b.append([vocabulary.get(char, len(vocabulary) + 1) for char in input_b if char not in stopwords])
                y.append(float(label))
                char_len += len(x_a[-1]) + len(x_b[-1])
                if len(x_a[-1]) > max_len:
                    max_len = len(x_a[-1])
                if len(x_b[-1]) > max_len:
                    max_len = len(x_b[-1])

        print('avg_char_len:', float(char_len) / (len(x_a)*2))
        print('max_char_len:', max_len)
        return x_a, x_b, y, vocabulary


def load_sentence(x, y):
    with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
        vocabulary = pickle.load(f_vocabulary)
    words_a = nltk.word_tokenize(x)
    words_b = nltk.word_tokenize(y)
    x_ = [vocabulary.get(word, len(vocabulary) + 1) for word in words_a if word not in stopwords]
    y_ = [vocabulary.get(word, len(vocabulary) + 1) for word in words_b if word not in stopwords]
    return x_, y_


if __name__ == '__main__':

    # vocab = build_word_level_vocabulary_all('../data/sent_train.txt', '../data/sent_dev.txt', '../data/sent_test.txt')
    # with open('../data/word_level/vocabulary_origin_all.pkl', 'wb') as vocabulary_pkl:
    #     pickle.dump(vocab, vocabulary_pkl, -1)
    #     print(len(vocab))

    # build_word_level_corpus_all('../data/sent_train.txt', '../data/sent_dev.txt', '../data/sent_test.txt')
    # 引入外部数据集
    # build_word_level_vocabulary_from_out_corpus('../raw_data/open_data/text.txt', '../data/word_level/text.txt')
    generate_embedding('word')
    # generate_fasttext_embedding('word')

    # vocab = build_char_level_vocabulary_all('data/xxx_train.jsonl', 'data/xxx_dev.jsonl', 'data/xxx_test.jsonl')
    # with open('data/char_level/vocabulary_all.pkl', 'wb') as vocabulary_pkl:
    #     pickle.dump(vocab, vocabulary_pkl, -1)
    #     print(len(vocab))
    # build_char_level_corpus_all('data/xxx_train.jsonl', 'data/xxx_dev.jsonl', 'data/xxx_test.jsonl')
    # generate_embedding('char')
    # # generate_fasttext_embedding('char')

