# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
import os
import datetime
from collections import Counter
import pickle

def set_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(2019)
    random.seed(2019)
    tf.set_random_seed(2019)

set_seed()

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cuda', '0', 'gpu id')
tf.app.flags.DEFINE_boolean('pre_embed', True, 'load pre-trained word2vec')
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
tf.app.flags.DEFINE_integer('epochs', 200, 'max train epochs')
tf.app.flags.DEFINE_integer('hidden_dim', 300, 'dimension of hidden embedding')
tf.app.flags.DEFINE_integer('word_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('pos_dim', 5, 'dimension of position embedding')
tf.app.flags.DEFINE_integer('pos_limit', 30, 'max distance of position embedding')
tf.app.flags.DEFINE_integer('sen_len', 60, 'sentence length')
tf.app.flags.DEFINE_integer('window', 3, 'window size')
tf.app.flags.DEFINE_string('model_path', './modfile', 'save model dir')
tf.app.flags.DEFINE_string('result_path', './results', 'save result dir')
tf.app.flags.DEFINE_string('data_path', './raw_data/open_data', 'data dir to load')
tf.app.flags.DEFINE_string('level', 'bag', 'bag level or sentence level, option:bag/sent')
tf.app.flags.DEFINE_string('model_name', 'cnn', 'model name')
tf.app.flags.DEFINE_string('loss_type', 'clf', 'loss type')
tf.app.flags.DEFINE_float('class_weight', 0.1, 'class weight')
tf.app.flags.DEFINE_string('mode', 'train', 'train or test')
tf.app.flags.DEFINE_boolean('lexical', False, 'add lexical feature')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout rate')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('word_frequency', 5, 'minimum word frequency when constructing vocabulary list')

class Baseline:
    def __init__(self, flags):
        self.lr = flags.lr
        self.sen_len = flags.sen_len
        self.pre_embed = flags.pre_embed
        self.pos_limit = flags.pos_limit
        self.pos_dim = flags.pos_dim
        self.window = flags.window
        self.word_dim = flags.word_dim
        self.hidden_dim = flags.hidden_dim
        self.batch_size = flags.batch_size
        self.data_path = flags.data_path
        self.model_path = flags.model_path
        self.result_path = flags.result_path
        self.mode = flags.mode
        self.epochs = flags.epochs
        self.dropout = flags.dropout
        self.word_frequency = flags.word_frequency
        self.cw = flags.class_weight
        self.model_name = flags.model_name
        self.loss_type = flags.loss_type
        self.pre_epochs = 1
        self.exp_name = '%s_%s' % (self.model_name, self.loss_type)
        if self.cw < 1:
            self.exp_name += '_cw'

        self.filter_lengths = [2, 3, 4, 5]
        self.lexical = flags.lexical

        if flags.level == 'sent':
            self.bag = False
            self.sent_bag = False
        elif flags.level == 'bag':
            self.bag = True
            self.sent_bag = False
        else:
            self.bag = False
            self.sent_bag = True
            self.exp_name += 'sent_bag'
        print(self.exp_name)

        self.pos_num = 2 * self.pos_limit + 3
        self.relation2id = self.load_relation()
        self.num_classes = len(self.relation2id)
        self.rnn_dropout_keep_prob = 0.7

        if self.loss_type == 'na_rl':
            self.num_classes -= 1

        if self.pre_embed:
            self.wordMap, word_embed = self.load_wordVec()
            self.word_embedding = tf.get_variable(initializer=word_embed, name='word_embedding', trainable=False)

        else:
            self.wordMap = self.load_wordMap()
            self.word_embedding = tf.get_variable(shape=[len(self.wordMap), self.word_dim], name='word_embedding',trainable=True)

        self.pos_e1_embedding = tf.get_variable(name='pos_e1_embedding', shape=[self.pos_num, self.pos_dim])
        self.pos_e2_embedding = tf.get_variable(name='pos_e2_embedding', shape=[self.pos_num, self.pos_dim])


        if self.model_name == 'cnn' or self.model_name == 'cnn_multi':
            self.sentence_reps = self.CNN_encoder()
        if self.model_name == 'textcnn':
            self.sentence_reps = self.TextCNN_encoder()
        if self.model_name == 'lstm':
            self.sentence_reps = self.LSTM_encoder()
        if self.model_name == 'attlstm':
            self.sentence_reps = self.AttLSTM_encoder()
        if self.model_name == 'pcnn':
            self.sentence_reps = self.PCNN_encoder()
        if self.model_name == 'mlcnn':
            self.sentence_reps = self.MLCNN_encoder()
        if self.model_name == 'attcnn':
            self.sent_dim = 300
        else:
            self.sent_dim = self.sentence_reps.get_shape()[-1]
        if self.loss_type == 'rl':
            import math
            r = math.sqrt(6.0 / (self.num_classes + self.hidden_dim))
            self.relation_embedding = tf.get_variable(name='relation_embedding',
                                                      shape=[self.sent_dim, self.num_classes],
                                                      initializer=tf.random_uniform_initializer(minval=-r, maxval=r))
        else:
            self.relation_embedding = tf.get_variable(name='relation_embedding',
                                                      shape=[self.sent_dim, self.num_classes])
        self.relation_embedding_b = tf.get_variable(name='relation_embedding_b', shape=[self.num_classes])

        if self.model_name == 'attcnn':
            self.sentence_reps = self.AttCNN_encoder()

        if self.bag:
            self.bag_level()
        elif self.sent_bag:
            self.sent_bag_level()
        else:
            self.sentence_level()
        self._classifier_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.classifier_loss)

    def pos_index(self, x):
        if x < -self.pos_limit:
            return 0
        if x >= -self.pos_limit and x <= self.pos_limit:
            return x + self.pos_limit + 1
        if x > self.pos_limit:
            return 2 * self.pos_limit + 2

    def load_wordVec(self):
        if not os.path.exists('data/word_level/vocabulary.pkl'):
            wordMap = {}
            wordMap['PAD'] = len(wordMap)
            wordMap['UNK'] = len(wordMap)
            word_embed = []
            for line in open(os.path.join(self.data_path, 'word2vec.txt')):
                content = line.strip().split()
                if len(content) != self.word_dim + 1:
                    continue
                wordMap[content[0]] = len(wordMap)
                word_embed.append(np.asarray(content[1:], dtype=np.float32))

            word_embed = np.stack(word_embed)
            embed_mean, embed_std = word_embed.mean(), word_embed.std()

            pad_embed = np.random.normal(embed_mean, embed_std, (2, self.word_dim))
            word_embed = np.concatenate((pad_embed, word_embed), axis=0)
            word_embed = word_embed.astype(np.float32)
            with open('data/word_level/vocabulary.pkl', 'wb') as vocabulary_pkl:
                pickle.dump(wordMap, vocabulary_pkl, -1)
                print(len(wordMap))
            np.save(open('data/word_level/ccks_300_dim.embeddings', 'wb'), word_embed)
        else:
            word_embed = np.load('data/word_level/ccks.embeddings')
            with open('data/word_level/vocabulary.pkl', 'rb') as f_vocabulary:
                wordMap = pickle.load(f_vocabulary, encoding='bytes')
            wordMap['PAD'] = 0
            wordMap['UNK'] = 1
        return wordMap, word_embed

    def load_wordMap(self):
        wordMap = {}
        wordMap['PAD'] = len(wordMap)
        wordMap['UNK'] = len(wordMap)
        all_content = []
        for line in open(os.path.join(self.data_path, 'sent_train.txt')):
            all_content += line.strip().split('\t')[3].split()
        for item in Counter(all_content).most_common():
            if item[1] > self.word_frequency:
                wordMap[item[0]] = len(wordMap)
            else:
                break
        return wordMap

    def load_relation(self):
        relation2id = {}
        for line in open(os.path.join(self.data_path, 'relation2id.txt')):
            relation, id_ = line.strip().split()
            relation2id[relation] = int(id_)
        return relation2id

    def load_sent(self, filename):
        sentence_dict = {}
        with open(os.path.join(self.data_path, filename), 'r') as fr:
            for line in fr:
                id_, en1, en2, sentence = line.strip().split('\t')
                sentence = sentence.split()
                en1_pos = 0
                en2_pos = 0
                for i in range(len(sentence)):
                    if sentence[i] == en1:
                        en1_pos = i
                    if sentence[i] == en2:
                        en2_pos = i
                words = []
                pos1 = []
                pos2 = []
                e1 = [0] * self.sen_len
                e1[en1_pos] = 1
                e2 = [0] * self.sen_len
                e2[en2_pos] = 1
                e_mask = []

                length = min(self.sen_len, len(sentence))

                for i in range(length):
                    words.append(self.wordMap.get(sentence[i], self.wordMap['UNK']))
                    pos1.append(self.pos_index(i - en1_pos))
                    pos2.append(self.pos_index(i - en2_pos))
                    if i <= en1_pos:
                        e_mask.append(0)
                    elif i > en1_pos and i <= en2_pos:
                        e_mask.append(1)
                    else:
                        e_mask.append(2)

                if length < self.sen_len:
                    for i in range(length, self.sen_len):
                        words.append(self.wordMap['PAD'])
                        pos1.append(self.pos_index(i - en1_pos))
                        pos2.append(self.pos_index(i - en2_pos))
                        e_mask.append(2)
                sentence_dict[id_] = np.reshape(np.asarray([words, pos1, pos2, e1, e2, e_mask], dtype=np.int32), (1, 6, self.sen_len))
        return sentence_dict

    def data_batcher(self, sentence_dict, filename, padding=False, shuffle=True):
        if self.bag:
            all_bags = []
            all_sents = []
            all_labels = []
            with open(os.path.join(self.data_path, filename), 'r') as fr:
                for line in fr:
                    if self.loss_type == 'na_rl':
                        rel = [0] * (self.num_classes + 1)
                    else:
                        rel = [0] * self.num_classes
                    try:
                        bag_id, _, _, sents, types = line.strip().split('\t')
                        type_list = types.split()
                        for tp in type_list:
                            if len(type_list) > 1 and tp == '0': # if a bag has multiple relations, we only consider non-NA relations
                                continue
                            rel[int(tp)] = 1
                    except:
                        bag_id, _, _, sents = line.strip().split('\t')

                    sent_list = []
                    for sent in sents.split():
                        sent_list.append(sentence_dict[sent])

                    all_bags.append(bag_id)
                    all_sents.append(np.concatenate(sent_list,axis=0))
                    all_labels.append(np.asarray(rel, dtype=np.float32))

            self.data_size = len(all_bags)
            self.datas = all_bags
            data_order = list(range(self.data_size))
            if shuffle:
                np.random.shuffle(data_order)
            if padding:
                if self.data_size % self.batch_size != 0:
                    data_order += [data_order[-1]] * (self.batch_size - self.data_size % self.batch_size)

            for i in range(len(data_order) // self.batch_size):
                total_sens = 0
                out_sents = []
                out_sent_nums = []
                out_labels = []
                for k in data_order[i * self.batch_size:(i + 1) * self.batch_size]:
                    out_sents.append(all_sents[k])
                    out_sent_nums.append(total_sens)
                    total_sens += all_sents[k].shape[0]
                    out_labels.append(all_labels[k])

                out_sents = np.concatenate(out_sents, axis=0)
                out_sent_nums.append(total_sens)
                out_sent_nums = np.asarray(out_sent_nums, dtype=np.int32)
                out_labels = np.stack(out_labels)

                yield out_sents, out_labels, out_sent_nums
        else:
            all_sent_ids = []
            all_sents = []
            all_labels = []
            with open(os.path.join(self.data_path, filename), 'r') as fr:
                for line in fr:
                    if self.loss_type == 'na_rl':
                        rel = [0] * (self.num_classes + 1)
                    else:
                        rel = [0] * self.num_classes
                    try:
                        sent_id, types = line.strip().split('\t')
                        type_list = types.split()
                        for tp in type_list:
                            if len(type_list) > 1 and tp == '0': # if a sentence has multiple relations, we only consider non-NA relations
                                continue
                            rel[int(tp)] = 1
                    except:
                        sent_id = line.strip()

                    all_sent_ids.append(sent_id)
                    all_sents.append(sentence_dict[sent_id])

                    if self.loss_type == 'na_rl':
                        all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, self.num_classes + 1)))
                    else:
                        all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, self.num_classes)))

            self.data_size = len(all_sent_ids)
            self.datas = all_sent_ids

            all_sents = np.concatenate(all_sents, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            data_order = list(range(self.data_size))
            if shuffle:
                np.random.shuffle(data_order)
            if padding:
                if self.data_size % self.batch_size != 0:
                    data_order += [data_order[-1]] * (self.batch_size - self.data_size % self.batch_size)

            for i in range(len(data_order) // self.batch_size):
                idx = data_order[i * self.batch_size:(i + 1) * self.batch_size]
                yield all_sents[idx], all_labels[idx], None

    def sent_bag_data_batcher(self, sentence_dict, filename, train=True, padding=False, shuffle=True):
        if train:
            all_bags = []
            all_sents = []
            all_labels = []
            with open(os.path.join(self.data_path, filename), 'r') as fr:
                for line in fr:
                    if self.loss_type == 'na_rl':
                        rel = [0] * (self.num_classes + 1)
                    else:
                        rel = [0] * self.num_classes
                    try:
                        bag_id, _, _, sents, types = line.strip().split('\t')
                        type_list = types.split()
                        for tp in type_list:
                            if len(type_list) > 1 and tp == '0': # if a bag has multiple relations, we only consider non-NA relations
                                continue
                            rel[int(tp)] = 1
                    except:
                        bag_id, _, _, sents = line.strip().split('\t')

                    sent_list = []
                    for sent in sents.split():
                        sent_list.append(sentence_dict[sent])

                    all_bags.append(bag_id)
                    all_sents.append(np.concatenate(sent_list,axis=0))
                    all_labels.append(np.asarray(rel, dtype=np.float32))

            self.data_size = len(all_bags)
            self.datas = all_bags
            data_order = list(range(self.data_size))
            if shuffle:
                np.random.shuffle(data_order)
            if padding:
                if self.data_size % self.batch_size != 0:
                    data_order += [data_order[-1]] * (self.batch_size - self.data_size % self.batch_size)

            for i in range(len(data_order) // self.batch_size):
                total_sens = 0
                out_sents = []
                out_sent_nums = []
                out_labels = []
                for k in data_order[i * self.batch_size:(i + 1) * self.batch_size]:
                    out_sents.append(all_sents[k])
                    out_sent_nums.append(total_sens)
                    total_sens += all_sents[k].shape[0]
                    out_labels.append(all_labels[k])

                out_sents = np.concatenate(out_sents, axis=0)
                out_sent_nums.append(total_sens)
                out_sent_nums = np.asarray(out_sent_nums, dtype=np.int32)
                out_labels = np.stack(out_labels)

                yield out_sents, out_labels, out_sent_nums
        else:
            all_sent_ids = []
            all_sents = []
            all_labels = []
            with open(os.path.join(self.data_path, filename), 'r') as fr:
                for line in fr:
                    if self.loss_type == 'na_rl':
                        rel = [0] * (self.num_classes + 1)
                    else:
                        rel = [0] * self.num_classes
                    try:
                        sent_id, types = line.strip().split('\t')
                        type_list = types.split()
                        for tp in type_list:
                            if len(type_list) > 1 and tp == '0': # if a sentence has multiple relations, we only consider non-NA relations
                                continue
                            rel[int(tp)] = 1
                    except:
                        sent_id = line.strip()

                    all_sent_ids.append(sent_id)
                    all_sents.append(sentence_dict[sent_id])

                    if self.loss_type == 'na_rl':
                        all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, self.num_classes + 1)))
                    else:
                        all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, self.num_classes)))

            self.data_size = len(all_sent_ids)
            self.datas = all_sent_ids

            all_sents = np.concatenate(all_sents, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            data_order = list(range(self.data_size))
            if shuffle:
                np.random.shuffle(data_order)
            if padding:
                if self.data_size % self.batch_size != 0:
                    data_order += [data_order[-1]] * (self.batch_size - self.data_size % self.batch_size)

            for i in range(len(data_order) // self.batch_size):
                idx = data_order[i * self.batch_size:(i + 1) * self.batch_size]
                # print np.asarray(range(self.batch_size + 1), dtype=np.int32)
                # print np.asarray(range(self.batch_size + 1), dtype=np.int32).shape
                yield all_sents[idx], all_labels[idx], np.asarray(range(self.batch_size + 1), dtype=np.int32)

    def CNN_encoder(self):
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_word')
        self.input_pos_e1 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e1')
        self.input_pos_e2 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e2')
        if self.loss_type == 'na_rl':
            self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes + 1], name='input_label')
        else:
            self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_label')

        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(self.word_embedding, self.input_word), \
                                                   tf.nn.embedding_lookup(self.pos_e1_embedding, self.input_pos_e1), \
                                                   tf.nn.embedding_lookup(self.pos_e2_embedding, self.input_pos_e2)])
        inputs_forward = tf.expand_dims(inputs_forward, -1)

        with tf.name_scope('conv-maxpool'):
            w = tf.get_variable(name='w', shape=[self.window, self.word_dim + 2 * self.pos_dim, 1, self.hidden_dim])
            b = tf.get_variable(name='b', shape=[self.hidden_dim])
            conv = tf.nn.conv2d(
                inputs_forward,
                w,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv')
            h = tf.nn.bias_add(conv, b)
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sen_len - self.window + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='pool')
        sen_reps = tf.tanh(tf.reshape(pooled, [-1, self.hidden_dim]))
        sen_reps = tf.nn.dropout(sen_reps, self.keep_prob)
        return sen_reps

    @staticmethod
    def splitting(i, p_subj_ind, p_obj_ind, bwc_conv, channels_count, outputs):
        l_ind = tf.minimum(tf.gather(p_subj_ind, [i]), tf.gather(p_obj_ind, [i]))  # left
        r_ind = tf.maximum(tf.gather(p_subj_ind, [i]), tf.gather(p_obj_ind, [i]))  # right

        l_ind = tf.cast(l_ind, dtype=tf.int32)
        r_ind = tf.cast(r_ind, dtype=tf.int32)
        w = tf.Variable(bwc_conv.shape[1], dtype=tf.int32)  # total width (words count)

        b_slice_from = [i, 0, 0]
        b_slice_size = tf.concat([[1], l_ind, [channels_count]], 0)
        m_slice_from = tf.concat([[i], l_ind, [0]], 0)
        m_slice_size = tf.concat([[1], r_ind - l_ind, [channels_count]], 0)
        a_slice_from = tf.concat([[i], r_ind, [0]], 0)
        a_slice_size = tf.concat([[1], w - r_ind, [channels_count]], 0)

        bwc_split_b = tf.slice(bwc_conv, b_slice_from, b_slice_size)
        bwc_split_m = tf.slice(bwc_conv, m_slice_from, m_slice_size)
        bwc_split_a = tf.slice(bwc_conv, a_slice_from, a_slice_size)

        pad_b = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([w - l_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        pad_m = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([w - r_ind + l_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        pad_a = tf.concat([[[0, 0]],
                           tf.reshape(tf.concat([r_ind, [0]], 0), shape=[1, 2]),
                           [[0, 0]]],
                          axis=0)

        # bwc_split_b = tf.pad(bwc_split_b, pad_b, constant_values=tf.float32.min)
        # bwc_split_m = tf.pad(bwc_split_m, pad_m, constant_values=tf.float32.min)
        # bwc_split_a = tf.pad(bwc_split_a, pad_a, constant_values=tf.float32.min)
        bwc_split_b = tf.pad(bwc_split_b, pad_b, constant_values=0)
        bwc_split_m = tf.pad(bwc_split_m, pad_m, constant_values=0)
        bwc_split_a = tf.pad(bwc_split_a, pad_a, constant_values=0)

        # outputs = outputs.write(i, tf.concat([bwc_split_b, bwc_split_m, bwc_split_a], axis=-1))
        outputs = outputs.write(i, [[bwc_split_b, bwc_split_m, bwc_split_a]])

        i += 1
        return i, p_subj_ind, p_obj_ind, bwc_conv, channels_count, outputs

    @staticmethod
    def padding(embedded_data, window_size):
        assert (isinstance(window_size, int) and window_size > 0)

        left_padding = (window_size - 1) / 2
        right_padding = (window_size - 1) - left_padding
        return tf.pad(embedded_data, [[0, 0],
                                      [left_padding, right_padding],
                                      [0, 0]])

    def PCNN_encoder(self):
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_word')
        self.input_pos_e1 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e1')
        self.input_pos_e2 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e2')
        self.input_e1 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_e1')
        self.input_e2 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_e2')
        input_e1 = tf.argmax(self.input_e1, axis=-1)
        input_e2 = tf.argmax(self.input_e2, axis=-1)
        if self.loss_type == 'na_rl':
            self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes + 1], name='input_label')
        else:
            self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_label')

        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(self.word_embedding, self.input_word), \
                                                   tf.nn.embedding_lookup(self.pos_e1_embedding, self.input_pos_e1), \
                                                   tf.nn.embedding_lookup(self.pos_e2_embedding, self.input_pos_e2)])

        w = tf.get_variable(name='w', shape=[self.window * (self.word_dim + 2 * self.pos_dim), 1, self.hidden_dim])
        b = tf.get_variable(name='b', shape=[self.hidden_dim])
        inputs_forward = self.padding(inputs_forward, self.window)

        inputs_forward_line = tf.reshape(inputs_forward, [self.batch_size, (self.sen_len + (self.window - 1)) *
                                                          (self.word_dim + 2 * self.pos_dim), 1])
        conv = tf.nn.conv1d(inputs_forward_line, w, self.word_dim + 2 * self.pos_dim,
                                padding="VALID",
                                name="conv")
        # conv = tf.nn.conv1d(
        #     inputs_forward, w, stride=1, padding='SAME', name='conv')
        h = tf.nn.bias_add(conv, b)

        sliced = tf.TensorArray(dtype=tf.float32, size=self.batch_size, infer_shape=False, dynamic_size=True)
        _, _, _, _, _, sliced = tf.while_loop(
            lambda i, *_: tf.less(i, self.batch_size),
            self.splitting,
            [0, input_e1, input_e2, h, self.hidden_dim, sliced])
        sliced = tf.squeeze(sliced.concat())
        # print(sliced.get_shape())
        sliced = tf.reshape(sliced, [self.batch_size, self.sen_len, 1, self.hidden_dim * 3])
        pooled = tf.nn.max_pool(sliced, ksize=[1, self.sen_len, 1, 1], strides=[1, self.sen_len, 1, 1],
                                padding='VALID', name='pool')

        # pooled = tf.squeeze(pooled, [2])
        # pooled = tf.transpose(pooled, perm=[0, 2, 1])

        sen_reps = tf.tanh(tf.reshape(pooled, [-1, self.hidden_dim * 3]))
        sen_reps = tf.nn.dropout(sen_reps, self.keep_prob)
        return sen_reps

    def MLCNN_encoder(self):
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_word')
        self.input_pos_e1 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e1')
        self.input_pos_e2 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e2')
        self.input_e1 = tf.placeholder(dtype=tf.float32, shape=[None, self.sen_len], name='input_e1')
        self.input_e2 = tf.placeholder(dtype=tf.float32, shape=[None, self.sen_len], name='input_e2')
        if self.loss_type == 'na_rl':
            self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes + 1], name='input_label')
        else:
            self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_label')

        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(self.word_embedding, self.input_word), \
                                                   tf.nn.embedding_lookup(self.pos_e1_embedding, self.input_pos_e1), \
                                                   tf.nn.embedding_lookup(self.pos_e2_embedding, self.input_pos_e2)])
        # inputs_forward = tf.expand_dims(inputs_forward, -1)

        e1_embedding = tf.matmul(tf.expand_dims(self.input_e1, axis=1), inputs_forward)
        e2_embedding = tf.matmul(tf.expand_dims(self.input_e2, axis=1), inputs_forward)

        a_e1 = tf.matmul(inputs_forward, tf.transpose(e1_embedding, [0, 2, 1]))
        a_e1 = tf.nn.softmax(a_e1, axis=1)
        a_e2 = tf.matmul(inputs_forward, tf.transpose(e2_embedding, [0, 2, 1]))
        a_e2 = tf.nn.softmax(a_e2, axis=1)
        a = (a_e1 + a_e2) / 2
        # inputs_forward = a * inputs_forward
        # inputs_forward = tf.expand_dims(inputs_forward, -1)

        with tf.name_scope('conv-maxpool'):
            # w = tf.get_variable(name='w', shape=[self.window, self.word_dim + 2 * self.pos_dim, 1, self.hidden_dim])
            w = tf.get_variable(name='w', shape=[self.window, self.word_dim + 2 * self.pos_dim, self.hidden_dim])
            b = tf.get_variable(name='b', shape=[self.hidden_dim])
            # conv = tf.nn.conv2d(
            #     inputs_forward,
            #     w,
            #     strides=[1, 1, 1, 1],
            #     padding='VALID',
            #     name='conv')
            conv = tf.nn.conv1d(inputs_forward, w, stride=1, padding='SAME', name='conv')
            h = tf.nn.bias_add(conv, b)
            # h = tf.expand_dims(h, axis=-2)
            pooled = tf.reduce_sum(h * a, axis=-2)
            # pooled = tf.nn.max_pool(
            #     h,
            #     ksize=[1, self.sen_len - self.window + 1, 1, 1],
            #     strides=[1, 1, 1, 1],
            #     padding='VALID',
            #     name='pool')
        sen_reps = tf.tanh(tf.reshape(pooled, [-1, self.hidden_dim]))
        sen_reps = tf.nn.dropout(sen_reps, self.keep_prob)
        # sen_reps = tf.concat([sen_reps, tf.squeeze(e1_embedding, axis=1), tf.squeeze(e2_embedding, axis=1)], axis=-1)
        return sen_reps

    def bag_level(self):

        self.class_weight = tf.placeholder(dtype=tf.float32, shape=[self.num_classes], name='class_weight')

        self.classifier_loss = 0.0
        self.probability = []

        self.bag_sens = tf.placeholder(dtype=tf.int32, shape=[self.batch_size + 1], name='bag_sens')
        self.att_A = tf.get_variable(name='att_A', shape=[self.hidden_dim])
        self.rel = tf.reshape(tf.transpose(self.relation_embedding), [self.num_classes, self.hidden_dim])  # (num_classes, hidden_dim)

        for i in range(self.batch_size):
            sen_reps = tf.reshape(self.sentence_reps[self.bag_sens[i]:self.bag_sens[i + 1]], [-1, self.hidden_dim])  #（sent_num, hidden_dim）

            att_sen = tf.reshape(tf.multiply(sen_reps, self.att_A), [-1, self.hidden_dim])  #（sent_num, hidden_dim）
            score = tf.matmul(self.rel, tf.transpose(att_sen))  # (num_classes, sent_num)
            alpha = tf.nn.softmax(score, 1)  #（num_classes, sent_num)
            bag_rep = tf.matmul(alpha, sen_reps)  #（num_classes, hidden_dim)

            out = tf.matmul(bag_rep, self.relation_embedding) + self.relation_embedding_b  # (num_classes, num_classes)

            # prob = tf.reshape(tf.reduce_sum(tf.nn.softmax(out, 1) * tf.reshape(self.input_label[i], [-1, 1]), 0),
            #                   [self.num_classes])

            self.probability.append(
                tf.reshape(tf.reduce_sum(tf.nn.softmax(out, 1) * tf.diag([1.0] * (self.num_classes)), 1),
                           [-1, self.num_classes]))
            self.classifier_loss += tf.reduce_sum(
                -tf.log(tf.clip_by_value(tf.reshape(self.probability[i], [self.num_classes]), 1.0e-10, 1.0)) * tf.reshape(self.input_label[i], [-1]))

        self.probability = tf.concat(axis=0, values=self.probability)
        self.classifier_loss = self.classifier_loss / tf.cast(self.batch_size, tf.float32)

    # def bag_level(self):
    #     self.classifier_loss = 0.0
    #     self.probability = []
    #
    #     self.bag_sens = tf.placeholder(dtype=tf.int32, shape=[self.batch_size + 1], name='bag_sens')
    #     self.att_A = tf.get_variable(name='att_A', shape=[self.hidden_dim])
    #     self.rel = tf.reshape(tf.transpose(self.relation_embedding), [self.num_classes, self.hidden_dim])  # (num_classes, hidden_dim)
    #
    #     for i in range(self.batch_size):
    #         sen_reps = tf.reshape(self.sentence_reps[self.bag_sens[i]:self.bag_sens[i + 1]], [-1, self.hidden_dim])  #（sent_num, hidden_dim）
    #
    #         att_sen = tf.reshape(tf.multiply(sen_reps, self.att_A), [-1, self.hidden_dim])  #（sent_num, hidden_dim）
    #         score = tf.matmul(self.rel, tf.transpose(att_sen))  # (num_classes, sent_num)
    #         alpha = tf.nn.softmax(score, 1)  #（num_classes, sent_num)
    #         bag_rep = tf.matmul(alpha, sen_reps)  #（num_classes, hidden_dim)
    #
    #         out = tf.matmul(bag_rep, self.relation_embedding) + self.relation_embedding_b  # (num_classes, num_classes)
    #
    #         prob = tf.reshape(tf.reduce_sum(tf.nn.softmax(out, 1) * tf.reshape(self.input_label[i], [-1, 1]), 0),
    #                           [self.num_classes])
    #
    #         self.probability.append(
    #             tf.reshape(tf.reduce_sum(tf.nn.softmax(out, 1) * tf.diag([1.0] * (self.num_classes)), 1),
    #                        [-1, self.num_classes]))
    #         self.classifier_loss += tf.reduce_sum(
    #             -tf.log(tf.clip_by_value(prob, 1.0e-10, 1.0)) * tf.reshape(self.input_label[i], [-1]))
    #
    #     self.probability = tf.concat(axis=0, values=self.probability)
    #     self.classifier_loss = self.classifier_loss / tf.cast(self.batch_size, tf.float32)

    def sent_bag_level(self):
        # self.classifier_loss = 0.0
        # self.probability = []
        #
        # self.class_weight = tf.placeholder(dtype=tf.float32, shape=[self.num_classes], name='class_weight')
        # self.bag_sens = tf.placeholder(dtype=tf.int32, shape=[self.batch_size + 1], name='bag_sens')
        # # self.att_A = tf.get_variable(name='att_A', shape=[self.hidden_dim])
        # # self.rel = tf.reshape(tf.transpose(self.relation_embedding), [self.num_classes, self.hidden_dim])
        #
        # for i in range(self.batch_size):
        #     sen_reps = tf.reshape(self.sentence_reps[self.bag_sens[i]:self.bag_sens[i + 1]], [-1, self.hidden_dim])
        #     score = tf.matmul(sen_reps, self.relation_embedding) + self.relation_embedding_b
        #     score = tf.nn.softmax(score, 1)
        #     prob = tf.reduce_sum(score, 0)
        #     self.probability.append(prob)
        #     self.classifier_loss += tf.reduce_sum(-tf.log(tf.clip_by_value(prob, 1.0e-10, 1.0)) * self.input_label[i] * self.class_weight)
        #
        # self.probability = tf.concat(axis=0, values=self.probability)
        # self.classifier_loss = self.classifier_loss / tf.cast(self.batch_size, tf.float32)
        self.classifier_loss = 0.0
        self.probability = []
        self.class_weight = tf.placeholder(dtype=tf.float32, shape=[self.num_classes], name='class_weight')

        self.bag_sens = tf.placeholder(dtype=tf.int32, shape=[self.batch_size + 1], name='bag_sens')
        self.att_A = tf.get_variable(name='att_A', shape=[self.hidden_dim])
        self.rel = tf.reshape(tf.transpose(self.relation_embedding), [self.num_classes, self.hidden_dim])

        for i in range(self.batch_size):
            sen_reps = tf.reshape(self.sentence_reps[self.bag_sens[i]:self.bag_sens[i + 1]], [-1, self.hidden_dim])

            att_sen = tf.reshape(tf.multiply(sen_reps, self.att_A), [-1, self.hidden_dim])
            score = tf.matmul(self.rel, tf.transpose(att_sen))
            alpha = tf.nn.softmax(score, 1)
            bag_rep = tf.matmul(alpha, sen_reps)

            out = tf.matmul(bag_rep, self.relation_embedding) + self.relation_embedding_b

            prob = tf.reshape(tf.reduce_sum(tf.nn.softmax(out, 1) * tf.reshape(self.input_label[i], [-1, 1]), 0),
                              [self.num_classes])

            self.probability.append(
                tf.reshape(tf.reduce_sum(tf.nn.softmax(out, 1) * tf.diag([1.0] * (self.num_classes)), 1),
                           [-1, self.num_classes]))
            self.classifier_loss += tf.reduce_sum(
                -tf.log(tf.clip_by_value(prob, 1.0e-10, 1.0)) * tf.reshape(self.input_label[i], [-1]) * self.class_weight)

        self.probability = tf.concat(axis=0, values=self.probability)
        self.classifier_loss = self.classifier_loss / tf.cast(self.batch_size, tf.float32)

    def sentence_level(self):
        if self.model_name == 'cnn_multi':
            self.class_weight = tf.placeholder(dtype=tf.float32, shape=[self.num_classes], name='class_weight')
            self.class_weight2 = tf.placeholder(dtype=tf.float32, shape=[2], name='class_weight')
            out = tf.matmul(self.sentence_reps, self.relation_embedding) + self.relation_embedding_b
            self.probability = tf.nn.softmax(out, 1)
            self.relation_embedding2 = tf.get_variable(name='relation_embedding_2',
                                                      shape=[self.sent_dim, 2])
            self.relation_embedding_b2 = tf.get_variable(name='relation_embedding_b_2', shape=[2])
            out2 = tf.matmul(self.sentence_reps, self.relation_embedding2) + self.relation_embedding_b2
            input_label2 = tf.one_hot(tf.cast(self.input_label[:, 0] < 1, dtype=tf.int32), 2)
            self.probability2 = tf.nn.softmax(out2, 1)
            self.classifier_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.clip_by_value(self.probability,
                                                                                         1.0e-10, 1.0)) *
                                                                self.input_label * self.class_weight, 1)) + \
                                   tf.reduce_mean(tf.reduce_sum( -tf.log(tf.clip_by_value(self.probability2,
                                                                                          1.0e-10, 1.0)) * input_label2
                                                                 * self.class_weight2, 1))

        if self.loss_type == 'clf':
            self.class_weight = tf.placeholder(dtype=tf.float32, shape=[self.num_classes], name='class_weight')
            out = tf.matmul(self.sentence_reps, self.relation_embedding) + self.relation_embedding_b
            self.probability = tf.nn.softmax(out, 1)
            # tf.ones([self.num_classes], dtype=tf.float32)
            self.classifier_loss = tf.reduce_mean(
                tf.reduce_sum(-tf.log(tf.clip_by_value(self.probability, 1.0e-10, 1.0)) * self.input_label * self.class_weight, 1))
        if self.loss_type == 'rl':
            m_pos = tf.constant(2.5)
            m_neg = tf.constant(0.5)
            r = tf.constant(2.0)
            s = tf.matmul(self.sentence_reps, self.relation_embedding) + self.relation_embedding_b
            self.probability = tf.nn.softmax(s, 1)
            s_pos = tf.reduce_sum(s * self.input_label, -1)
            _, index = tf.nn.top_k(s, k=2)

            L = tf.constant(0.0)
            i = tf.constant(0)
            cond = lambda i, L: tf.less(i, self.batch_size)

            def loop_body(i, L):
                s_neg_i = tf.cond(tf.equal(self.input_label[i, index[i][0]], 1),
                                 lambda: index[i][1], lambda: index[i][0])
                # s_neg_i = tf.cond(tf.logical_and(tf.equal(self.input_label[i, index[i][0]], 1),
                #                                  tf.equal(self.input_label[i, index[i][1]], 1)),
                #                   lambda: index[i][2], lambda: s_neg_i)

                s_neg = s[i, s_neg_i]

                l = tf.log((1.0 + tf.exp((r * (m_pos - s_pos[i]))))) + \
                    tf.log((1.0 + tf.exp((r * (m_neg + s_neg)))))

                return [tf.add(i, 1), tf.add(L, l)]

            _, L = tf.while_loop(cond, loop_body, loop_vars=[i, L])

            vars_ = [v for v in tf.trainable_variables() if 'relation_embedding_b' not in v.name
                     and 'word_embedding' not in v.name and 'pos_e1_embedding' not in v.name and 'pos_e2_embedding' not in v.name]
            l2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in vars_]), 0.001)

            nbatch = tf.to_float(self.batch_size)
            self.classifier_loss = L / nbatch + l2_loss
        if self.loss_type == 'na_rl':
            m_pos = tf.constant(2.5)
            m_neg = tf.constant(0.5)
            r = tf.constant(2.0)
            s = tf.matmul(self.sentence_reps, self.relation_embedding) + self.relation_embedding_b
            # self.probability = tf.nn.softmax(s, 1)
            self.probability = s
            s_pos = tf.reduce_sum(s * self.input_label[:, 1:], -1)
            _, index = tf.nn.top_k(s, k=2)
            y_na = 1 - self.input_label[:, 0]

            L = tf.constant(0.0)
            i = tf.constant(0)
            cond = lambda i, L: tf.less(i, self.batch_size)

            def loop_body(i, L):
                s_neg_i = tf.cond(tf.equal(self.input_label[i, index[i][0] + 1], 1),
                                  lambda: index[i][1], lambda: index[i][0])
                # s_neg_i = tf.cond(tf.logical_and(tf.equal(self.input_label[i, index[i][0] + 1], 1),
                #                                  tf.equal(self.input_label[i, index[i][1] + 1], 1)),
                #                   lambda: index[i][2], lambda: s_neg_i)

                s_neg = s[i, s_neg_i]

                l = y_na[i] * tf.log((1.0 + tf.exp((r * (m_pos - s_pos[i]))))) + \
                    tf.log((1.0 + tf.exp((r * (m_neg + s_neg)))))

                return [tf.add(i, 1), tf.add(L, l)]

            _, L = tf.while_loop(cond, loop_body, loop_vars=[i, L])

            vars_ = [v for v in tf.trainable_variables() if 'relation_embedding_b' not in v.name
                     and 'word_embedding' not in v.name and 'pos_e1_embedding' not in v.name and 'pos_e2_embedding' not in v.name]
            l2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in vars_]), 0.001)

            nbatch = tf.to_float(self.batch_size)
            self.classifier_loss = L / nbatch + l2_loss
        if self.loss_type == 'pl':
            size = self.sentence_reps[0].get_shape()[-1]
            U = tf.get_variable(name='U', shape=[size, self.sent_dim])
            score = []

            for i in range(self.num_classes):
                score.append(tf.matmul(tf.matmul(self.sentence_reps[i], U),
                                                  tf.expand_dims(self.relation_embedding[:, i], axis=-1)))
            score = tf.concat(score, axis=-1)
            self.probability = tf.nn.softmax(score, 1)

            # m = tf.constant(1.0)
            #
            # L = tf.constant(0.0)
            # i = tf.constant(0)
            # cond = lambda i, L: tf.less(i, self.batch_size)
            #
            # def loop_body(i, L):
            #     s_pos_i = tf.cast(tf.argmax(self.input_label[i, :], -1), dtype=tf.int32)
            #     s_pos = score[i, s_pos_i]
            #     _, index = tf.nn.top_k(score[i, :], k=2)
            #     s_neg_i = tf.cond(tf.equal(s_pos_i, index[0]),
            #                       lambda: index[1], lambda: index[0])
            #     s_neg = score[i, s_neg_i]
            #
            #     l = tf.log((1.0 + tf.exp(m - s_pos + s_neg)))
            #
            #     return [tf.add(i, 1), tf.add(L, l)]
            #
            # _, L = tf.while_loop(cond, loop_body, loop_vars=[i, L])

            m_pos = tf.constant(2.5)
            m_neg = tf.constant(0.5)
            r = tf.constant(2.0)

            s_pos = tf.reduce_sum(score * self.input_label, -1)
            _, index = tf.nn.top_k(score, k=2)

            L = tf.constant(0.0)
            i = tf.constant(0)
            cond = lambda i, L: tf.less(i, self.batch_size)

            def loop_body(i, L):
                s_neg_i = tf.cond(tf.equal(self.input_label[i, index[i][0]], 1),
                                  lambda: index[i][1], lambda: index[i][0])
                # s_neg_i = tf.cond(tf.logical_and(tf.equal(self.input_label[i, index[i][0]], 1),
                #                                  tf.equal(self.input_label[i, index[i][1]], 1)),
                #                   lambda: index[i][2], lambda: s_neg_i)

                s_neg = score[i, s_neg_i]

                l = tf.log((1.0 + tf.exp((r * (m_pos - s_pos[i]))))) + \
                    tf.log((1.0 + tf.exp((r * (m_neg + s_neg)))))

                return [tf.add(i, 1), tf.add(L, l)]

            _, L = tf.while_loop(cond, loop_body, loop_vars=[i, L])

            vars_ = [v for v in tf.trainable_variables() if 'relation_embedding_b' not in v.name
                     and 'word_embedding' not in v.name and 'pos_e1_embedding' not in v.name and 'pos_e2_embedding' not in v.name]
            l2_loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in vars_]), 0.001)

            nbatch = tf.to_float(self.batch_size)
            self.classifier_loss = L / nbatch + l2_loss

    def run_train(self, sess, batch):

        sent_batch, label_batch, sen_num_batch = batch

        feed_dict = {}
        feed_dict[self.keep_prob] = self.dropout
        feed_dict[self.input_word] = sent_batch[:, 0, :]
        feed_dict[self.input_pos_e1] = sent_batch[:, 1, :]
        feed_dict[self.input_pos_e2] = sent_batch[:, 2, :]
        cw = [1] * self.num_classes
        cw2 = [1, 1]
        if self.cw < 1:
            cw[0] = self.cw
            cw2[0] = self.cw
        feed_dict[self.class_weight] = np.asarray(cw)
        if self.model_name[-5:] == 'multi':
            feed_dict[self.class_weight2] = np.asarray(cw2)
        if self.model_name == 'pcnn':
            feed_dict[self.input_e1] = sent_batch[:, 3, :]
            feed_dict[self.input_e2] = sent_batch[:, 4, :]
        # if self.model_name == 'pcnn':
        #     feed_dict[self.input_e_mask] = sent_batch[:, 5, :]
        feed_dict[self.input_label] = label_batch
        if self.bag or self.sent_bag:
            feed_dict[self.bag_sens] = sen_num_batch

        _, classifier_loss = sess.run([self._classifier_train_op, self.classifier_loss], feed_dict)

        return classifier_loss

    def run_dev(self, sess, dev_batchers):
        all_labels = []
        all_probs = []
        for batch in dev_batchers:
            sent_batch, label_batch, sen_num_batch = batch
            all_labels.append(label_batch)

            feed_dict = {}
            feed_dict[self.keep_prob] = 1.0
            feed_dict[self.input_word] = sent_batch[:, 0, :]
            feed_dict[self.input_pos_e1] = sent_batch[:, 1, :]
            feed_dict[self.input_pos_e2] = sent_batch[:, 2, :]
            if self.model_name == 'pcnn':
                feed_dict[self.input_e1] = sent_batch[:, 3, :]
                feed_dict[self.input_e2] = sent_batch[:, 4, :]
            # if self.model_name == 'pcnn':
            #     feed_dict[self.input_e_mask] = sent_batch[:, 5, :]
            if self.bag or self.sent_bag:
                feed_dict[self.bag_sens] = sen_num_batch
            prob = sess.run([self.probability], feed_dict)
            all_probs.append(np.reshape(prob, (-1, self.num_classes)))

        all_labels = np.concatenate(all_labels, axis=0)[:self.data_size]
        all_probs = np.concatenate(all_probs, axis=0)[:self.data_size]
        if self.bag:
            all_preds = all_probs
            all_preds[all_probs > 0.9] = 1
            all_preds[all_probs <= 0.9] = 0
        else:
            if self.loss_type == 'na_rl':
                all_preds = np.zeros((all_probs.shape[0], self.num_classes + 1))
                all_neg = np.sum(all_probs < 0, axis=-1)
                for i in range(all_preds.shape[0]):
                    if all_neg[i] == self.num_classes:
                        all_preds[i, 0] = 1
                    else:
                        all_preds[i, np.argmax(all_probs[i], axis=-1) + 1] = 1
            else:
                all_preds = np.eye(self.num_classes)[np.reshape(np.argmax(all_probs, 1), (-1))]

        return all_preds, all_labels

    def run_test(self, sess, test_batchers):
        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)
        all_probs = []
        for batch in test_batchers:
            sent_batch, _, sen_num_batch = batch

            feed_dict = {}
            feed_dict[self.keep_prob] = 1.0
            feed_dict[self.input_word] = sent_batch[:, 0, :]
            feed_dict[self.input_pos_e1] = sent_batch[:, 1, :]
            feed_dict[self.input_pos_e2] = sent_batch[:, 2, :]
            if self.model_name == 'pcnn':
                feed_dict[self.input_e1] = sent_batch[:, 3, :]
                feed_dict[self.input_e2] = sent_batch[:, 4, :]
            # if self.model_name == 'pcnn':
            #     feed_dict[self.input_e_mask] = sent_batch[:, 5, :]
            if self.bag or self.sent_bag:
                feed_dict[self.bag_sens] = sen_num_batch
            prob = sess.run([self.probability], feed_dict)
            all_probs.append(np.reshape(prob, (-1, self.num_classes)))

        all_probs = np.concatenate(all_probs,axis=0)[:self.data_size]
        if self.bag:
            all_preds = all_probs
            all_preds[all_probs > 0.9] = 1
            all_preds[all_probs <= 0.9] = 0
        else:
            # all_preds = np.eye(self.num_classes)[np.reshape(np.argmax(all_probs, 1), (-1))]
            if self.loss_type == 'na_rl':
                all_preds = np.zeros((all_probs.shape[0], self.num_classes + 1))
                all_neg = np.sum(all_probs < 0, axis=-1)
                for i in range(all_preds.shape[0]):
                    if all_neg[i] == self.num_classes:
                        all_preds[i, 0] = 1
                    else:
                        all_preds[i, np.argmax(all_probs[i], axis=-1) + 1] = 1
            else:
                all_preds = np.eye(self.num_classes)[np.reshape(np.argmax(all_probs, 1), (-1))]

        if self.bag:
            with open(os.path.join(self.result_path, '%s_result_bag.txt' % self.exp_name), 'w') as fw:
                for i in range(self.data_size):
                    rel_one_hot = [int(num) for num in all_preds[i].tolist()]
                    rel_list = []
                    for j in range(0, self.num_classes):
                        if rel_one_hot[j] == 1:
                            rel_list.append(str(j))
                    if len(rel_list) == 0: # if a bag has no relation, it will be consider as having a relation NA
                        rel_list.append('0')
                    fw.write(self.datas[i] + '\t' + ' '.join(rel_list) + '\n')
        else:
            with open(os.path.join(self.result_path, '%s_result_sent.txt' % self.exp_name), 'w') as fw:
                for i in range(self.data_size):
                    rel_one_hot = [int(num) for num in all_preds[i].tolist()]
                    rel_list = []
                    if self.loss_type == 'na_rl':
                        nc = self.num_classes + 1
                    else:
                        nc = self.num_classes
                    for j in range(0, nc):
                        if rel_one_hot[j] == 1:
                            rel_list.append(str(j))
                    fw.write(self.datas[i] + '\t' + ' '.join(rel_list) + '\n')

    def run_model(self, sess, saver):
        if self.mode == 'train':
            global_step = 0
            sent_train = self.load_sent('sent_train.txt')
            sent_dev = self.load_sent('sent_dev.txt')
            max_f1 = 0.0
            max_epoch = 0

            if not os.path.isdir(self.model_path):
                os.mkdir(self.model_path)

            if not os.path.isdir(os.path.join(self.model_path, self.exp_name)):
                os.mkdir(os.path.join(self.model_path, self.exp_name))

            for epoch in range(self.epochs):
                if self.bag:
                    train_batchers = self.data_batcher(sent_train, 'bag_relation_train.txt', padding=False, shuffle=True)
                elif self.sent_bag:
                    train_batchers = self.sent_bag_data_batcher(sent_train, 'bag_relation_train.txt', padding=False, shuffle=True)
                else:
                    train_batchers = self.data_batcher(sent_train, 'sent_relation_train.txt', padding=False,
                                                       shuffle=True)
                for batch in train_batchers:

                    losses = self.run_train(sess, batch)
                    global_step += 1
                    if global_step % 50 == 0:
                        time_str = datetime.datetime.now().isoformat()
                        tempstr = "{}: step {}, classifier_loss {:g}".format(time_str, global_step, losses)
                        print(tempstr)
                    if global_step % 200 == 0:
                        print(epoch, global_step)
                        if self.bag:
                            dev_batchers = self.data_batcher(sent_dev, 'bag_relation_dev.txt', padding=True, shuffle=False)
                        elif self.sent_bag:
                            dev_batchers = self.sent_bag_data_batcher(sent_dev, 'sent_relation_dev.txt', train=False, padding=True, shuffle=False)
                        else:
                            dev_batchers = self.data_batcher(sent_dev, 'sent_relation_dev.txt', padding=True, shuffle=False)
                        all_preds, all_labels = self.run_dev(sess, dev_batchers)

                        # when calculate f1 score, we don't consider whether NA results are predicted or not
                        # the number of non-NA answers in test is counted as n_std
                        # the number of non-NA answers in predicted answers is counted as n_sys
                        # intersection of two answers is counted as n_r
                        n_r = int(np.sum(all_preds[:, 1:] * all_labels[:, 1:]))
                        n_std = int(np.sum(all_labels[:,1:]))
                        n_sys = int(np.sum(all_preds[:,1:]))
                        try:
                            precision = float(n_r) / float(n_sys)
                            recall = float(n_r) / float(n_std)
                            f1 = 2 * precision * recall / (precision + recall)
                        except ZeroDivisionError:
                            f1 = 0.0
                        print(f1, '   max_f1:', max_f1, '  max_epoch', max_epoch)

                        if f1 > max_f1:
                            max_f1 = f1
                            max_epoch = epoch
                            print('f1: %f' % f1)
                            print('saving model')
                            if self.sent_bag:
                                path = saver.save(sess, os.path.join(self.model_path, self.exp_name,
                                                                     'ipre_sent_bag_%d' % (self.bag)), global_step=0)
                            else:
                                path = saver.save(sess, os.path.join(self.model_path, self.exp_name, 'ipre_bag_%d' % (self.bag)), global_step=0)
                            tempstr = 'have saved model to ' + path
                            print(tempstr)
            print("max_f1:", max_f1, '  max_epoch', max_epoch)
        elif self.mode == 'pre_train':
            path = os.path.join(self.model_path, self.exp_name, 'ipre_sent_bag_%d' % (self.bag)) + '-0'
            tempstr = 'load model: ' + path
            print(tempstr)
            try:
                saver.restore(sess, path)
            except:
                raise ValueError('Unvalid model name')

            global_step = 0
            sent_dev = self.load_sent('sent_dev.txt')
            max_f1 = 0.0
            max_epoch = 0

            if not os.path.isdir(self.model_path):
                os.mkdir(self.model_path)

            if not os.path.isdir(os.path.join(self.model_path, self.exp_name)):
                os.mkdir(os.path.join(self.model_path, self.exp_name))

            for epoch in range(self.pre_epochs):
                if self.bag:
                    train_batchers = self.data_batcher(sent_dev, 'bag_relation_dev.txt', padding=False, shuffle=True)
                elif self.sent_bag:
                    train_batchers = self.sent_bag_data_batcher(sent_dev, 'sent_relation_dev.txt', train=False, padding=False, shuffle=True)
                else:
                    train_batchers = self.data_batcher(sent_dev, 'sent_relation_dev.txt', padding=False,
                                                       shuffle=True)
                for batch in train_batchers:

                    losses = self.run_train(sess, batch)
                    global_step += 1
                    if global_step % 50 == 0:
                        time_str = datetime.datetime.now().isoformat()
                        tempstr = "{}: step {}, classifier_loss {:g}".format(time_str, global_step, losses)
                        print(tempstr)
                    if global_step % 200 == 0:
                        print(epoch, global_step)
                        if self.bag:
                            dev_batchers = self.data_batcher(sent_dev, 'bag_relation_dev.txt', padding=True, shuffle=False)
                        elif self.sent_bag:
                            dev_batchers = self.sent_bag_data_batcher(sent_dev, 'sent_relation_dev.txt', train=False, padding=True, shuffle=False)
                        else:
                            dev_batchers = self.data_batcher(sent_dev, 'sent_relation_dev.txt', padding=True, shuffle=False)
                        all_preds, all_labels = self.run_dev(sess, dev_batchers)

                        # when calculate f1 score, we don't consider whether NA results are predicted or not
                        # the number of non-NA answers in test is counted as n_std
                        # the number of non-NA answers in predicted answers is counted as n_sys
                        # intersection of two answers is counted as n_r
                        n_r = int(np.sum(all_preds[:, 1:] * all_labels[:, 1:]))
                        n_std = int(np.sum(all_labels[:,1:]))
                        n_sys = int(np.sum(all_preds[:,1:]))
                        try:
                            precision = float(n_r) / float(n_sys)
                            recall = float(n_r) / float(n_std)
                            f1 = 2 * precision * recall / (precision + recall)
                        except ZeroDivisionError:
                            f1 = 0.0
                        print(f1, '   max_f1:', max_f1, '  max_epoch', max_epoch)

                        if f1 > max_f1:
                            max_f1 = f1
                            max_epoch = epoch
                            print('f1: %f' % f1)
                            print('saving model')
                            path = saver.save(sess, os.path.join(self.model_path, self.exp_name, 'ipre_sent_bag_dev'), global_step=0)
                            tempstr = 'have saved model to ' + path
                            print(tempstr)
            print("max_f1:", max_f1, '  max_epoch', max_epoch)
        else:
            if self.sent_bag:
                path = os.path.join(self.model_path, self.exp_name, 'ipre_sent_bag_dev') + '-0'
            else:
                path = os.path.join(self.model_path, self.exp_name, 'ipre_bag_%d' % (self.bag)) + '-0'
            tempstr = 'load model: ' + path
            print(tempstr)
            try:
                saver.restore(sess, path)
            except:
                raise ValueError('Unvalid model name')

            sent_test = self.load_sent('sent_test.txt')
            if self.bag:
                test_batchers = self.data_batcher(sent_test, 'bag_relation_test.txt', padding=True, shuffle=False)
            elif self.sent_bag:
                test_batchers = self.sent_bag_data_batcher(sent_test, 'sent_relation_test.txt', train=False, padding=True, shuffle=False)
            else:
                test_batchers = self.data_batcher(sent_test, 'sent_relation_test.txt', padding=True, shuffle=False)

            self.run_test(sess, test_batchers)


def get_word2vec():
    # -*- coding: utf-8 -*-
    from gensim.models import word2vec
    from gensim.models.word2vec import LineSentence
    import logging, sys
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    sentences = word2vec.Text8Corpus(os.path.join('data/word_level', 'text.txt'))  # 加载语料
    # sentences = LineSentence('data/word_level/text_corpus.txt')
    model = word2vec.Word2Vec(sentences, sg=1, size=300, window=5, min_count=10, negative=5, sample=1e-4,
                              workers=10)
    model.wv.save_word2vec_format('data/word2vec.txt', binary=False)

def word():
    if not os.path.exists('data/word_level/vocabulary.pkl'):
        wordMap = {}
        wordMap['PAD'] = len(wordMap)
        wordMap['UNK'] = len(wordMap)
        word_embed = []
        for line in open('data/word2vec.txt'):
            content = line.strip().split()
            if len(content) != 300 + 1:
                continue
            # print type(content[0])
            wordMap[content[0].decode('utf-8')] = len(wordMap)
            word_embed.append(np.asarray(content[1:], dtype=np.float32))

        # word_embed = np.stack(word_embed)
        # embed_mean, embed_std = word_embed.mean(), word_embed.std()
        #
        # pad_embed = np.random.normal(embed_mean, embed_std, (2, 300))
        # word_embed = np.concatenate((pad_embed, word_embed), axis=0)
        # word_embed = word_embed.astype(np.float32)
        with open('data/word_level/vocabulary.pkl', 'wb') as vocabulary_pkl:
            pickle.dump(wordMap, vocabulary_pkl, -1)
            print(len(wordMap))
        # np.save(open('data/word_level/ccks_300_dim.embeddings', 'wb'), word_embed)
    else:
        word_embed = np.load('data/word_level/ccks_300_dim.embeddings')
        with open('data/word_level/vocabulary.pkl', 'rb') as f_vocabulary:
            wordMap = pickle.load(f_vocabulary)
    return wordMap, word_embed


def main(_):
    tf.reset_default_graph()
    print('build model')
    gpu_options = tf.GPUOptions(visible_device_list=FLAGS.cuda, allow_growth=True)
    with tf.Graph().as_default():
        set_seed()
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('', initializer=initializer):
                model = Baseline(FLAGS)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
            model.run_model(sess, saver)


if __name__ == '__main__':
    # get_word2vec()
    tf.app.run()
