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
from sklearn.feature_extraction.text import TfidfVectorizer

random.seed(42)

stopwords = [u'', u' ', '\t', '.', u',', '=']
gold_label = {'entails': 1, 'neutral': 0}


# 训练外部词向量得到embedding
def generate_embedding(level):
    data_path = '../data/%s_level' % level
    save_model_file = '../modfile/Word2Vec.mod'
    save_model_name = '../modfile/sst_300_dim_all.embeddings'
    word_size = 300

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 输入的语料是用jieba分好词的文本
    sentences = word2vec.Text8Corpus(os.path.join(data_path, 'text.txt'))  # 加载语料
    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
    model = word2vec.Word2Vec(sentences, min_count=1, size=word_size, window=5, workers=4)
    vocab = pickle.load(open(os.path.join(data_path, 'vocabulary_all.pkl'), 'rb'))
    weights = model.wv.syn0
    # 得到词向量字典
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 2, word_size), dtype='float32')
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_file, binary=False)
    # vocab 形式： {word : index}
    for w, i in vocab.items():
        if w not in d:
            continue
        emb[i, :] = weights[d[w], :]
    np.save(open(save_model_name, 'wb'), emb)


# 训练原始语料中的词向量得到embedding
def generate_embedding_raw(level):
    data_path = '../data/%s_level' % level
    save_model_file = '../modfile/Raw_Word2Vec.mod'
    save_model_name = '../modfile/raw_sst_300_dim_all.embeddings'
    word_size = 300

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 输入的语料是用jieba分好词的文本
    sentences = word2vec.Text8Corpus(os.path.join(data_path, 'corpus_raw.txt'))  # 加载语料
    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
    model = word2vec.Word2Vec(sentences, min_count=1, size=word_size, window=5, workers=4)
    model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_name, binary=False)  # 以二进制类型保存模型以便重用

    vocab = pickle.load(open(os.path.join(data_path, 'vocabulary_all.pkl'), 'rb'))
    weights = model.wv.syn0
    # 得到词向量字典
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 2, word_size), dtype='float32')
    model.save('../modfile/Raw_Word2Vec.mod')
    model.wv.save_word2vec_format('../modfile/Raw_Word2Vec.mod', binary=False)
    # vocab 形式： {word : index}
    for w, i in vocab.items():
        if w not in d:
            continue
        emb[i, :] = weights[d[w], :]
    np.save(open('../modfile/raw_sst_300_dim_all.embeddings', 'wb'), emb)


# 训练集、验证集划分
def train_valid_split(raw_file):
    labels = list()

    with codecs.open(raw_file, encoding='utf-8') as f_raw:
        lines = f_raw.readlines()
        for line in lines:
            line = json.loads(line)
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

    target_lines = [' '.join([w for w in sentence.split(' ') if w not in stopwords]).lower() + '\n' for sentence in
                    sentences]

    with codecs.open('../data/word_level/raw_corpus_all.txt', 'w', encoding='utf-8') as f_corpus:
        f_corpus.writelines(target_lines)


# 建立字符级别的语料，目前暂时不用
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


# 生成词典，获取训练集，验证集，测试集的单词
def build_word_level_vocabulary_all(train_file, valid_file, test_file):
    sentences = list()
    with codecs.open(train_file, encoding='utf-8') as f_train:
        lines = f_train.readlines()
        for line in lines:
            json_data = json.loads(line)
            # 对句子进行分词
            sentences.extend(json_data['sent'].split(' '))

    with codecs.open(valid_file, encoding='utf-8') as f_valid:
        lines = f_valid.readlines()
        for line in lines:
            json_data = json.loads(line)
            sentences.extend(json_data['sent'].split(' '))

    with codecs.open(test_file, encoding='utf-8') as f_test:
        lines = f_test.readlines()
        for line in lines:
            json_data = json.loads(line)
            sentences.extend(json_data['sent'].split(' '))

    # 转换成集合，去掉重复词
    word_list = list(set(sentences))
    print(len(word_list))
    word_list = [word for word in word_list if word not in stopwords]
    # 转换成字典并返回
    return dict((word, idx + 1) for idx, word in enumerate(word_list))


# 从外部语料中建立单词级别的语料
def build_word_level_vocabulary_from_out_corpus(raw_file, out_file):
    sentences = list()
    with codecs.open(raw_file, encoding='utf-8') as f_test:
        lines = f_test.readlines()
        for line in lines:
            # 分词
            sentence_cut = jieba.cut(line)
            # 转换成词向量训练时的格式，单词与单词之间空格分隔
            sentences.append(' '.join(sentence_cut))

    # 保存到外部文件中，方便训练词向量使用
    with codecs.open(out_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(sentences)


# 字级别的语料，这里暂时不用
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

    return dict((char, idx + 1) for idx, char in enumerate(char_list))


# 获得标签
def load_label(raw_file):
    y = list()
    with codecs.open(raw_file, encoding='utf-8') as f_train:
        lines = f_train.readlines()
        for line in lines:
            input_a, input_b, label = line.strip().split('\t')
            y.append(float(label))
    return y


# 获取原始数据
def load_raw_data(raw_file, test=False):
    with open('data/word_level/vocabulary.pkl', 'rb') as f_vocabulary:
        vocabulary = pickle.load(f_vocabulary)
    if test:
        x = list()
        with codecs.open(raw_file, encoding='utf-8') as f_test:
            lines = f_test.readlines()
            for line in lines:
                json_data = json.loads(line)
                x.append([word for word in jieba.cut(json_data['sent'])])
        return x, vocabulary
    else:
        x = list()
        y = list()
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            for line in lines:
                json_data = json.loads(line)
                x.append([word for word in jieba.cut(json_data['sent'])])
                label = json_data['label']
                # 31 4这种标签单独处理，只保留第一个
                if len(label.split(' ')) > 1:
                    label = label.split(' ')[0]
                y.append(float(label))
        return x, y, vocabulary


# 根据不同的文件类型加载数据
def load_data(raw_file, level):
    # 字符级别的训练集和验证集
    if level == 'word':
        with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_word:', len(vocabulary))
        x = list()  # 句子输入
        disinfos1 = list()  # 人物1距离坐标输入
        disinfos2 = list()  # 人物2距离坐标输入
        y = list()  # 预测标签的输出
        max_len = 0
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            print(lines[0])
            for line in tqdm(lines):
                json_data = json.loads(line)
                input = json_data['sent']
                label = json_data['label']
                # 31 4这种标签单独处理
                if len(label.split(' ')) > 1:
                    label = label.split(' ')[0]

                # 单词中加入人物关系坐标
                per1 = json_data['per1']
                per2 = json_data['per2']

                words = input.split(' ')
                disinfo1 = load_distance(words, per1)
                disinfo2 = load_distance(words, per2)

                x.append([vocabulary.get(word, len(vocabulary) + 1) for word in words if word not in stopwords])
                y.append(float(label))
                # index -> vector
                disinfos1.append(disinfo1)
                disinfos2.append(disinfo2)

                if len(x[-1]) > max_len:
                    max_len = len(x[-1])
        print('max_word_len', max_len)
        avg_len = 0
        max_len = 0
        for word, id in vocabulary.items():
            if len(word) > max_len:
                max_len = len(word)
            avg_len += len(word)
        print('word_max_len:', max_len)
        print('word_avg_len:', float(avg_len) / len(vocabulary))
        return x, disinfos1, disinfos2, y, vocabulary

    # 测试集数据加载
    elif level == 'test':
        with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_word:', len(vocabulary))
        x = list()  # 句子输入
        disinfos1 = list()  # 人物1距离坐标输入
        disinfos2 = list()  # 人物2距离坐标输入
        ids = list()
        max_len = 0
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            print(lines[0])
            for line in tqdm(lines):
                json_data = json.loads(line)
                input = json_data['sent']
                test_id = json_data['id'].strip('\"')
                ids.append(test_id)
                words = input.split(' ')
                x.append([vocabulary.get(word, len(vocabulary) + 1) for word in words if word not in stopwords])

                # 单词中加入人物关系坐标
                per1 = json_data['per1']
                per2 = json_data['per2']
                disinfo1 = load_distance(words, per1)
                disinfo2 = load_distance(words, per2)
                disinfos1.append(disinfo1)
                disinfos2.append(disinfo2)

                if len(x[-1]) > max_len:
                    max_len = len(x[-1])
        print('max_word_len', max_len)
        avg_len = 0
        max_len = 0
        for word, id in vocabulary.items():
            if len(word) > max_len:
                max_len = len(word)
            avg_len += len(word)
        print('char_max_len:', max_len)
        print('char_avg_len:', float(avg_len) / len(vocabulary))
        return ids, x, disinfos1, disinfos2, vocabulary


# 根据不同的文件类型加载数据
def load_bag_data(raw_file, level):
    # 字符级别的训练集和验证集
    if level == 'word':
        with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_word:', len(vocabulary))
        x = list()  # 句子输入
        disinfos1 = list()  # 人物1距离坐标输入
        disinfos2 = list()  # 人物2距离坐标输入
        y = list()  # 预测标签的输出
        max_len = 0
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            print(lines[0])
            for line in tqdm(lines):
                json_data = json.loads(line)
                input = json_data['sent']
                label = json_data['label']
                # 31 4这种标签单独处理
                if len(label.split(' ')) > 1:
                    label = label.split(' ')[0]

                # 单词中加入人物关系坐标
                per1 = json_data['per1']
                per2 = json_data['per2']
                sents = input.split('@@@')
                x_sent = list()
                x_dissent1 = []
                x_dissent2 = []
                for sent in sents:
                    words = sent.split(' ')
                    disinfo1 = load_distance(words, per1)
                    disinfo2 = load_distance(words, per2)
                    x_sent.append([vocabulary.get(word, len(vocabulary) + 1) for word in words if word not in stopwords])
                    x_dissent1.append(disinfo1)
                    x_dissent2.append(disinfo2)
                x.append(x_sent)
                y.append(float(label))
                # index -> vector
                disinfos1.append(x_dissent1)
                disinfos2.append(x_dissent2)

                if len(x[-1]) > max_len:
                    max_len = len(x[-1])
        print('max_word_len', max_len)
        avg_len = 0
        max_len = 0
        for word, id in vocabulary.items():
            if len(word) > max_len:
                max_len = len(word)
            avg_len += len(word)
        print('word_max_len:', max_len)
        print('word_avg_len:', float(avg_len) / len(vocabulary))
        return x, disinfos1, disinfos2, y, vocabulary

    # 测试集数据加载
    elif level == 'test':
        with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_word:', len(vocabulary))
        x = list()  # 句子输入
        disinfos1 = list()  # 人物1距离坐标输入
        disinfos2 = list()  # 人物2距离坐标输入
        ids = list()
        max_len = 0
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            print(lines[0])
            for line in tqdm(lines):
                json_data = json.loads(line)
                input = json_data['sent']
                test_id = json_data['id'].strip('\"')
                ids.append(test_id)
                words = input.split(' ')
                x.append([vocabulary.get(word, len(vocabulary) + 1) for word in words if word not in stopwords])

                # 单词中加入人物关系坐标
                per1 = json_data['per1']
                per2 = json_data['per2']
                sents = input.split('@@@')
                x_sent = list()
                x_dissent1 = []
                x_dissent2 = []
                for sent in sents:
                    words = sent.split(' ')
                    disinfo1 = load_distance(words, per1)
                    disinfo2 = load_distance(words, per2)
                    x_sent.append(
                        [vocabulary.get(word, len(vocabulary) + 1) for word in words if word not in stopwords])
                    x_dissent1.append(disinfo1)
                    x_dissent2.append(disinfo2)
                x.append(x_sent)
                # index -> vector
                disinfos1.append(x_dissent1)
                disinfos2.append(x_dissent2)

                if len(x[-1]) > max_len:
                    max_len = len(x[-1])
        print('max_word_len', max_len)
        avg_len = 0
        max_len = 0
        for word, id in vocabulary.items():
            if len(word) > max_len:
                max_len = len(word)
            avg_len += len(word)
        print('char_max_len:', max_len)
        print('char_avg_len:', float(avg_len) / len(vocabulary))
        return ids, x, disinfos1, disinfos2, vocabulary


# 加载tf_idf数据
def load_tf_idf_data(raw_file, level):
    # 字符级别的训练集和验证集
    if level == 'word':
        x = list()  # 句子输入
        y = list()  # 预测标签的输出
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            print(lines[0])
            for line in tqdm(lines):
                json_data = json.loads(line)
                input = json_data['sent']
                label = json_data['label']
                # 31 4这种标签单独处理
                if len(label.split(' ')) > 1:
                    label = label.split(' ')[0]

                x.append(input)
                y.append(float(label))

        vectorizer = TfidfVectorizer(stop_words=stopwords,
                                     analyzer='word',
                                     min_df=5,
                                     token_pattern=r"(?u)\b\w+\b")
        tfidf = vectorizer.fit_transform(x)
        weight = tfidf.toarray()
        return weight, y

    # 测试集数据加载
    elif level == 'test':
        x = list()  # 句子输入
        ids = list()
        max_len = 0
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            print(lines[0])
            for line in tqdm(lines):
                json_data = json.loads(line)
                input = json_data['sent']

                x.append(input)

                if len(x[-1]) > max_len:
                    max_len = len(x[-1])
        vectorizer = TfidfVectorizer(stop_words=stopwords,
                                     analyzer='word',
                                     min_df=5,
                                     token_pattern=r"(?u)\b\w+\b")
        tfidf = vectorizer.fit_transform(x)
        weight = tfidf.toarray()
        return ids, weight


# 根据不同的文件类型加载数据
def load_data_multi_dis(raw_file, level):
    # 字符级别的训练集和验证集
    if level == 'word':
        with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_word:', len(vocabulary))
        x = list()  # 句子输入
        disinfos1 = list()  # 人物1距离坐标输入
        disinfos2 = list()  # 人物2距离坐标输入
        y = list()  # 预测标签的输出
        y2 = list()  # 二级标签的输出
        max_len = 0
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            print(lines[0])
            for line in tqdm(lines):
                json_data = json.loads(line)
                input = json_data['sent']
                label = json_data['label']
                label2 = json_data['label2']
                # 31 4这种标签单独处理
                if len(label.split(' ')) > 1:
                    label = label.split(' ')[0]

                # 单词中加入人物关系坐标
                per1 = json_data['per1']
                per2 = json_data['per2']

                words = input.split(' ')
                disinfo1 = load_distance(words, per1)
                disinfo2 = load_distance(words, per2)

                x.append([vocabulary.get(word, len(vocabulary) + 1) for word in words if word not in stopwords])
                y.append(float(label))
                y2.append(float(label2))
                # index -> vector
                disinfos1.append(disinfo1)
                disinfos2.append(disinfo2)

                if len(x[-1]) > max_len:
                    max_len = len(x[-1])
        print('max_word_len', max_len)
        avg_len = 0
        max_len = 0
        for word, id in vocabulary.items():
            if len(word) > max_len:
                max_len = len(word)
            avg_len += len(word)
        print('word_max_len:', max_len)
        print('word_avg_len:', float(avg_len) / len(vocabulary))
        return x, disinfos1, disinfos2, y, y2, vocabulary

    # 测试集数据加载
    elif level == 'test':
        with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_word:', len(vocabulary))
        x = list()  # 句子输入
        disinfos1 = list()  # 人物1距离坐标输入
        disinfos2 = list()  # 人物2距离坐标输入
        ids = list()
        max_len = 0
        with codecs.open(raw_file, encoding='utf-8') as f_train:
            lines = f_train.readlines()
            print(lines[0])
            for line in tqdm(lines):
                json_data = json.loads(line)
                input = json_data['sent']
                test_id = json_data['id'].strip('\"')
                ids.append(test_id)
                words = input.split(' ')
                x.append([vocabulary.get(word, len(vocabulary) + 1) for word in words if word not in stopwords])

                # 单词中加入人物关系坐标
                per1 = json_data['per1']
                per2 = json_data['per2']
                disinfo1 = load_distance(words, per1)
                disinfo2 = load_distance(words, per2)
                disinfos1.append(disinfo1)
                disinfos2.append(disinfo2)

                if len(x[-1]) > max_len:
                    max_len = len(x[-1])
        print('max_word_len', max_len)
        avg_len = 0
        max_len = 0
        for word, id in vocabulary.items():
            if len(word) > max_len:
                max_len = len(word)
            avg_len += len(word)
        print('char_max_len:', max_len)
        print('char_avg_len:', float(avg_len) / len(vocabulary))
        return ids, x, disinfos1, disinfos2, vocabulary


# 加入坐标距离
def load_distance(words, per):
    disinfo = np.arange(len(words))
    per_position = 0
    for word in words:
        if per == word:
            break
        else:
            per_position += 1
    position = np.array([per_position] * len(words))
    # 60 是句子的最大长度，防止索引中存在负值
    # 改进，句子中出现长度大于15的距离全部按照15
    limit_dis = disinfo - position
    # for i in range(len(words)):
    #     if abs(limit_dis[i]) > 15:
    #         limit_dis[i] = 15
    return limit_dis + 60


if __name__ == '__main__':
    vocab = build_word_level_vocabulary_all('../data/sent_train.txt', '../data/sent_dev.txt', '../data/sent_test.txt')
    with open('../data/word_level/vocabulary_all.pkl', 'wb') as vocabulary_pkl:
        pickle.dump(vocab, vocabulary_pkl, -1)
        print(len(vocab))

    # 这次暂时不用，本来外部语料就很多了
    # build_word_level_corpus_all('../data/sent_train.txt', '../data/sent_dev.txt', '../data/sent_test.txt')

    # 引入外部数据集
    # build_word_level_vocabulary_from_out_corpus('../raw_data/open_data/text.txt', '../data/word_level/text.txt')
    # 训练词向量
    generate_embedding('word')
    # generate_embedding_raw('word')
