# -*- encoding:utf-8 -*-

from keras.engine import Input
from keras.layers import Embedding, Dropout, Conv1D, Conv2D, MaxPooling2D, Lambda, LSTM, CuDNNLSTM, Dense, concatenate, \
    TimeDistributed, Bidirectional, GlobalMaxPool1D, GlobalAvgPool1D, Reshape, Flatten, Activation, BatchNormalization, \
    Dot, Permute, Multiply, Add, RepeatVector, GlobalMaxPooling1D, MaxPooling1D, ZeroPadding1D, SpatialDropout1D
from keras.activations import softmax
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from models.callbacks import distance_metrics, categorical_metrics, categorical_metrics_multi, categorical_metrics_multi_dis
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from keras.utils import to_categorical

import os
import numpy as np
import codecs


def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def manhattan_distance(vectors):
    x, y = vectors
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def acc(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


class Models(object):
    def __init__(self, config):
        self.config = config

    def init_callbacks(self):

        self.callbacks.append(categorical_metrics)
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            )
        )
        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.early_stopping_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.early_stopping_mode
            )
        )

    def init_callbacks_multi(self):

        self.callbacks.append(categorical_metrics_multi)
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            )
        )
        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.early_stopping_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.early_stopping_mode
            )
        )

    def init_callbacks_multi_dis(self):
        self.config.exp_name = 'bilstm_multi_dis_' + self.config.level
        self.callbacks.append(categorical_metrics_multi_dis)
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name),
                monitor=self.config.checkpoint_monitor,
                mode=self.config.checkpoint_mode,
                save_best_only=self.config.checkpoint_save_best_only,
                save_weights_only=self.config.checkpoint_save_weights_only,
                verbose=self.config.checkpoint_verbose,
            )
        )
        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.early_stopping_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.early_stopping_mode
            )
        )

    def load_weight(self, by_name=False):
        if self.model is None:
            raise Exception("You have to build the model first.")
        checkpoint_path = os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name)
        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path, by_name=by_name)
        print("Model loaded")

    # def siamese_cnn(self, distance=False, manhattan=False):
    #     weights = np.load(
    #         os.path.join(self.config.embedding_path, self.config.level + '_level', self.config.embedding_file))
    #     sentence1 = Input(shape=(self.config.max_len,), dtype='int32', name='sent1_base')
    #     sentence2 = Input(shape=(self.config.max_len,), dtype='int32', name='sent2_base')
    #     features = Input(shape=(self.config.features_len,), dtype='float32', name='features')
    #     embedding_layer = Embedding(input_dim=self.config.vocab_len + 2,
    #                                 output_dim=self.config.embedding_dim,
    #                                 weights=[weights], name='embedding_layer', trainable=True)
    #     sent1_embedding = embedding_layer(sentence1)
    #     sent2_embedding = embedding_layer(sentence2)
    #
    #     filter_lengths = [2, 3, 4, 5]
    #     sent1_conv_layers = []
    #     sent2_conv_layers = []
    #     for filter_length in filter_lengths:
    #         conv_layer = Conv1D(filters=200, kernel_size=filter_length, padding='valid',
    #                             activation='relu', strides=1)
    #         sent1_c = conv_layer(sent1_embedding)
    #         sent2_c = conv_layer(sent2_embedding)
    #         sent1_maxpooling = MaxPooling1D(pool_size=self.config.max_len - filter_length + 1)(sent1_c)
    #         sent1_flatten = Flatten()(sent1_maxpooling)
    #         sent1_conv_layers.append(sent1_flatten)
    #         sent2_maxpooling = MaxPooling1D(pool_size=self.config.max_len - filter_length + 1)(sent2_c)
    #         sent2_flatten = Flatten()(sent2_maxpooling)
    #         sent2_conv_layers.append(sent2_flatten)
    #     sent1_conv = concatenate(inputs=sent1_conv_layers)
    #     sent2_conv = concatenate(inputs=sent2_conv_layers)
    #
    #     if distance:
    #         distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([sent1_conv, sent2_conv])
    #         self.model = Model([sentence1, sentence2], distance)
    #         self.model.compile(loss=contrastive_loss, optimizer=self.config.optimizer, metrics=[acc])
    #     elif manhattan:
    #         distance = Lambda(manhattan_distance, output_shape=eucl_dist_output_shape)([sent1_conv, sent2_conv])
    #         self.model = Model([sentence1, sentence2], distance)
    #         self.model.compile(loss='binary_crossentropy',
    #                            optimizer=self.config.optimizer,
    #                            metrics=['acc'])
    #     else:
    #         sent = concatenate([sent1_conv, sent2_conv])
    #         sent = BatchNormalization(name='sent_representation')(sent)
    #         if self.config.features_len > 0:
    #             sent = concatenate([sent, features], axis=-1)
    #         x = Dense(400, activation='relu', name='dense1')(sent)
    #         x = BatchNormalization(name='batch_normal2')(x)
    #         x = Dense(400, activation='relu', name='dense2')(x)
    #         x = BatchNormalization(name='batch_normal3')(x)
    #         # simi = Dense(3, activation='softmax')(x)
    #         simi = Dense(1, activation='sigmoid', name='simi')(x)
    #
    #         if self.config.features_len == 0:
    #             self.model = Model(inputs=[sentence1, sentence2], outputs=simi)
    #         else:
    #             self.model = Model(inputs=[sentence1, sentence2, features], outputs=simi)
    #         self.model.compile(loss='binary_crossentropy', optimizer=self.config.optimizer,
    #                            metrics=['binary_accuracy'])
    #
    # def siamese_att_cnn(self, distance=False, manhattan=False):
    #     weights = np.load(
    #         os.path.join(self.config.embedding_path, self.config.level + '_level', self.config.embedding_file))
    #     sentence1 = Input(shape=(self.config.max_len,), dtype='int32', name='sent1_base')
    #     sentence2 = Input(shape=(self.config.max_len,), dtype='int32', name='sent2_base')
    #     features = Input(shape=(self.config.features_len,), dtype='float32', name='features')
    #     embedding_layer = Embedding(input_dim=self.config.vocab_len + 2,
    #                                 output_dim=self.config.embedding_dim,
    #                                 weights=[weights], name='embedding_layer', trainable=True)
    #     sent1_embedding = embedding_layer(sentence1)
    #     sent2_embedding = embedding_layer(sentence2)
    #
    #     attention = Dot(axes=-1)([sent1_embedding, sent2_embedding])
    #     wb = Lambda(lambda x: softmax(x, axis=1), output_shape=lambda x: x)(attention)
    #     wa = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2), output_shape=lambda x: x)(attention))
    #     sent1_ = Dot(axes=1)([wa, sent2_embedding])
    #     sent2_ = Dot(axes=1)([wb, sent1_embedding])
    #     neg = Lambda(lambda x: -x, output_shape=lambda x: x)
    #     substract1 = Add()([sent1_embedding, neg(sent1_)])
    #     mutiply1 = Multiply()([sent1_embedding, sent1_])
    #     substract2 = Add()([sent2_embedding, neg(sent2_)])
    #     mutiply2 = Multiply()([sent2_embedding, sent2_])
    #
    #     sent1_att = concatenate([sent1_, substract1, mutiply1])
    #     sent2_att = concatenate([sent2_, substract2, mutiply2])
    #
    #     filter_lengths = [2, 3, 4, 5]
    #     sent1_conv_layers = []
    #     sent2_conv_layers = []
    #     for filter_length in filter_lengths:
    #         conv_layer = AttConv1D(filters=200, kernel_size=filter_length, padding='same',
    #                                activation='relu', strides=1)
    #         sent1_conv = conv_layer([sent1_embedding, sent1_att])
    #         sent2_conv = conv_layer([sent2_embedding, sent2_att])
    #         sent1_conv = MaxPooling1D(pool_size=self.config.max_len)(sent1_conv)
    #         flatten = Flatten()(sent1_conv)
    #         sent1_conv_layers.append(flatten)
    #         sent2_conv = MaxPooling1D(pool_size=self.config.max_len)(sent2_conv)
    #         flatten = Flatten()(sent2_conv)
    #         sent2_conv_layers.append(flatten)
    #     sent1_conv = concatenate(inputs=sent1_conv_layers)
    #     sent2_conv = concatenate(inputs=sent2_conv_layers)
    #
    #     if distance:
    #         distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([sent1_conv, sent2_conv])
    #         self.model = Model([sentence1, sentence2], distance)
    #         self.model.compile(loss=contrastive_loss, optimizer=self.config.optimizer, metrics=[acc])
    #     elif manhattan:
    #         distance = Lambda(manhattan_distance, output_shape=eucl_dist_output_shape)([sent1_conv, sent2_conv])
    #         self.model = Model([sentence1, sentence2], distance)
    #         self.model.compile(loss='binary_crossentropy',
    #                            optimizer=self.config.optimizer,
    #                            metrics=['acc'])
    #     else:
    #         sent = concatenate([sent1_conv, sent2_conv])
    #         sent = BatchNormalization(name='sent_representation')(sent)
    #         if self.config.features_len > 0:
    #             sent = concatenate([sent, features], axis=-1)
    #         x = Dense(400, activation='relu', name='dense1')(sent)
    #         x = BatchNormalization(name='batch_normal2')(x)
    #         x = Dense(400, activation='relu', name='dense2')(x)
    #         x = BatchNormalization(name='batch_normal3')(x)
    #         # simi = Dense(3, activation='softmax')(x)
    #         simi = Dense(1, activation='sigmoid', name='simi')(x)
    #
    #         if self.config.features_len == 0:
    #             self.model = Model(inputs=[sentence1, sentence2], outputs=simi)
    #         else:
    #             self.model = Model(inputs=[sentence1, sentence2, features], outputs=simi)
    #         self.model.compile(loss='binary_crossentropy', optimizer=self.config.optimizer,
    #                            metrics=['binary_accuracy'])
    #
    # def cnn(self, pos=False, pi=False):
    #     sentence = Input(shape=(self.config.max_len,), dtype='int32', name='sent_base')
    #     features = Input(shape=(self.config.features_len,), dtype='float32', name='features')
    #     weights = np.load(os.path.join(self.config.embedding_path, 'word_level', self.config.embedding_file))
    #     if pi:
    #         p = np.zeros(shape=(4, weights.shape[-1]), dtype='float32')
    #         weights = np.vstack((weights, p))
    #     embedding_layer = Embedding(input_dim=weights.shape[0],
    #                                 output_dim=weights.shape[-1],
    #                                 weights=[weights], name='embedding_layer', trainable=True)
    #     sent_embedding = embedding_layer(sentence)
    #     if self.config.level == 'word_char':
    #         sentence_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32',
    #                               name='sent_char_base')
    #         weights_char = np.load(os.path.join(self.config.embedding_path, 'char_level', self.config.embedding_file))
    #         char_emb = self.char_embedding(weights_char)
    #         sent_char_embedding = char_emb(sentence_char)
    #         sent_embedding = concatenate([sent_embedding, sent_char_embedding])
    #     if pos:
    #         pos1 = Input(shape=(self.config.max_len,), dtype='int32', name='pos1')
    #         pos2 = Input(shape=(self.config.max_len,), dtype='int32', name='pos2')
    #
    #         pos_layer = Embedding(input_dim=2 * self.config.pos_limit + 3, output_dim=self.config.position_dim,
    #                               name='position_embedding', trainable=True)
    #         sent_pos1 = pos_layer(pos1)
    #         sent_pos2 = pos_layer(pos2)
    #         sent_embedding = concatenate([sent_embedding, sent_pos1, sent_pos2])
    #
    #     filter_length = 3
    #     conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='valid', strides=1)
    #     sent_c = conv_layer(sent_embedding)
    #     sent_maxpooling = MaxPooling1D(pool_size=self.config.max_len - filter_length + 1)(sent_c)
    #     sent_conv = Flatten()(sent_maxpooling)
    #     sent_conv = Activation('tanh')(sent_conv)
    #     sent = Dropout(0.5)(sent_conv)
    #     # sent = BatchNormalization(name='sent_representation')(sent_conv)
    #
    #     sent = concatenate([sent, features], axis=-1)
    #     # x = Dense(400, activation='relu')(sent)
    #     # x = BatchNormalization()(x)
    #     # x = Dense(400, activation='relu')(x)
    #     # x = BatchNormalization()(x)
    #     # output = Dense(35, activation='softmax', name='output')(x)
    #     output = Dense(35, activation='softmax', name='output')(sent)
    #
    #     # model = Model(inputs=[sentence1, sentence2], outputs=simi)
    #     inputs = [sentence]
    #     if self.config.level == 'word_char':
    #         inputs.append(sentence_char)
    #     # if pos:
    #     #     inputs.append(pos1)
    #     #     inputs.append(pos2)
    #     if self.config.features_len > 0:
    #         inputs.append(features)
    #     self.model = Model(inputs=inputs, outputs=output)
    #     self.model.compile(loss='categorical_crossentropy', optimizer=self.config.optimizer,
    #                            metrics=['accuracy'])

    # cnn基本demo
    def cnn_base(self):
        sentence = Input(shape=(self.config.max_len,), dtype='int32', name='sent_base')
        weights = np.load(os.path.join(self.config.embedding_path, self.config.embedding_file))
        # trainable修改为False
        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=False)
        sent_embedding = embedding_layer(sentence)
        filter_length = 3
        conv_layer = Conv1D(filters=100, kernel_size=filter_length, padding='valid', strides=1, activation='relu')
        sent_c = conv_layer(sent_embedding)
        sent_maxpooling = MaxPooling1D(pool_size=self.config.max_len - filter_length + 1)(sent_c)
        sent_conv = Flatten()(sent_maxpooling)
        sent_conv = Activation('relu')(sent_conv)
        sent = Dropout(0.5)(sent_conv)
        output = Dense(self.config.classes, activation='softmax', name='output')(sent)

        inputs = [sentence]
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.config.optimizer,
                           metrics=['acc'])

    # 双向lstm基本demo
    def bilstm_base(self):
        sentence = Input(shape=(self.config.max_len,), dtype='int32', name='sent_base')
        weights = np.load(os.path.join(self.config.embedding_path, self.config.embedding_file))
        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=True)
        sent_embedding = embedding_layer(sentence)
        sent_embedding = SpatialDropout1D(0.2)(sent_embedding)
        bilstm_layer = Bidirectional(LSTM(128))(sent_embedding)

        sent = Dropout(0.5)(bilstm_layer)
        output = Dense(self.config.classes, activation='softmax', name='output')(sent)
        inputs = [sentence]
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.config.optimizer,
                           metrics=['acc'])

    # 多任务模型
    def cnn_multi(self):
        sentence = Input(shape=(self.config.max_len,), dtype='int32', name='sent_base')
        weights = np.load(os.path.join(self.config.embedding_path, self.config.embedding_file))
        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=True)
        sent_embedding = embedding_layer(sentence)
        filter_length = 3
        conv_layer = Conv1D(filters=100, kernel_size=filter_length, padding='valid', strides=1, activation='relu')
        sent_c = conv_layer(sent_embedding)
        sent_maxpooling = MaxPooling1D(pool_size=self.config.max_len - filter_length + 1)(sent_c)
        sent_conv = Flatten()(sent_maxpooling)
        sent_conv = Activation('relu')(sent_conv)
        sent = Dropout(0.5)(sent_conv)
        # 多任务输出
        output = Dense(self.config.classes, activation='softmax', name='output')(sent)
        output2 = Dense(self.config.classes_multi, activation='softmax', name='output2')(sent)

        inputs = [sentence]
        outputs = [output, output2]
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss={'output': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'},
                           optimizer=self.config.optimizer,
                           loss_weights={'output': 1., 'output2': 1.},
                           metrics={'output': ['acc'], 'output2': ['acc']})

    # 多任务模型加入距离信息
    def cnn_multi_dis(self):
        sentence = Input(shape=(self.config.max_len,), dtype='int32', name='sent_base')
        dis1 = Input(shape=(self.config.max_len, 1), dtype='float32', name='disinfos1')
        dis2 = Input(shape=(self.config.max_len, 1), dtype='float32', name='disinfos2')
        weights = np.load(os.path.join(self.config.embedding_path, self.config.embedding_file))
        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=True)
        sent_embedding = embedding_layer(sentence)
        all_input = concatenate([sent_embedding, dis1, dis2], axis=2)
        filter_length = 3
        conv_layer = Conv1D(filters=100, kernel_size=filter_length, padding='valid', strides=1, activation='relu')
        sent_c = conv_layer(all_input)
        sent_maxpooling = MaxPooling1D(pool_size=self.config.max_len - filter_length + 1)(sent_c)
        sent_conv = Flatten()(sent_maxpooling)
        sent_conv = Activation('relu')(sent_conv)
        sent = Dropout(0.5)(sent_conv)
        # 多任务输出
        output = Dense(self.config.classes, activation='softmax', name='output')(sent)
        output2 = Dense(self.config.classes_multi, activation='softmax', name='output2')(sent)

        inputs = [sentence, dis1, dis2]
        outputs = [output, output2]
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss={'output': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'},
                           optimizer=self.config.optimizer,
                           loss_weights={'output': 1., 'output2': 1.},
                           metrics={'output': ['acc'], 'output2': ['acc']})

    # 多任务模型加入距离信息
    def bilstm_multi_dis(self):
        sentence = Input(shape=(self.config.max_len,), dtype='int32', name='sent_base')
        dis1 = Input(shape=(self.config.max_len, 1), dtype='float32', name='disinfos1')
        dis2 = Input(shape=(self.config.max_len, 1), dtype='float32', name='disinfos2')
        weights = np.load(os.path.join(self.config.embedding_path, self.config.embedding_file))
        # trainable修改为False
        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=False)

        embedding_dis1_layer = Embedding(input_dim=self.config.max_len*2,
                                         output_dim=5,
                                         name='embedding_dis1_layer', trainable=True)

        embedding_dis2_layer = Embedding(input_dim=self.config.max_len*2,
                                         output_dim=5,
                                         name='embedding_dis2_layer', trainable=True)

        sent_embedding = embedding_layer(sentence)
        dis1_embedding = embedding_dis1_layer(dis1)
        dis2_emdedding = embedding_dis2_layer(dis2)
        all_input = concatenate([sent_embedding, dis1_embedding, dis2_emdedding], axis=2)
        bilstm_layer = Bidirectional(LSTM(128))(all_input)
        sent = Dropout(0.5)(bilstm_layer)
        # 多任务输出
        output = Dense(self.config.classes, activation='softmax', name='output')(sent)
        output2 = Dense(self.config.classes_multi, activation='softmax', name='output2')(sent)

        inputs = [sentence, dis1, dis2]
        outputs = [output, output2]
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss={'output': 'categorical_crossentropy', 'output2': 'categorical_crossentropy'},
                           optimizer=self.config.optimizer,
                           loss_weights={'output': 1., 'output2': 1.},
                           metrics={'output': ['acc'], 'output2': ['acc']})

    def pad(self, x_data):
        return pad_sequences(x_data, maxlen=self.config.max_len, padding='post', truncating='post')

    def fit(self, x_train, y_train, x_valid, y_valid, train_features=None,
            valid_features=None):
        x_train = self.pad(x_train)
        x_valid = self.pad(x_valid)

        # 结果集one-hot，不能直接使用数字作为标签
        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)

        y_train = np.asarray(y_train)
        y_valid = np.asarray(y_valid)

        self.callbacks = []
        if train_features is not None:
            train_features = np.asarray(train_features)
            self.init_callbacks()
            self.model.fit([x_train, train_features], y_train,
                           epochs=self.config.num_epochs,
                           verbose=self.config.verbose_training,
                           batch_size=self.config.batch_size,
                           # 这里随机分出一部分数据作为验证集
                           validation_split=0.3,
                           callbacks=self.callbacks)
        else:
            # 初始化回调函数并用其训练
            self.init_callbacks()
            self.model.fit(x_train, y_train,
                           epochs=self.config.num_epochs,
                           verbose=self.config.verbose_training,
                           batch_size=self.config.batch_size,
                           validation_data=(x_valid, y_valid),
                           callbacks=self.callbacks,
                           class_weight='balanced')

    def fit_multi(self, x_train, y_train, y_train2, x_valid, y_valid, y_valid2, train_features=None,
                  valid_features=None):
        x_train = self.pad(x_train)
        x_valid = self.pad(x_valid)

        # 结果集one-hot，不能直接使用数字作为标签
        y_train = to_categorical(y_train)
        y_train2 = to_categorical(y_train2)
        y_valid = to_categorical(y_valid)
        y_valid2 = to_categorical(y_valid2)

        # 初始化回调函数并用其训练
        self.init_callbacks_multi()
        self.model.fit(x_train, [y_train, y_train2],
                       epochs=self.config.num_epochs,
                       verbose=self.config.verbose_training,
                       batch_size=self.config.batch_size,
                       validation_data=(x_valid, [y_valid, y_valid2]),
                       callbacks=self.callbacks)

    def fit_multi_dis(self, x_train, x_train_dis1, x_train_dis2, y_train, y_train2, x_valid, x_valid_dis1, x_valid_dis2, y_valid, y_valid2):
        x_train = self.pad(x_train)
        x_train_dis1 = np.array(x_train_dis1)
        x_train_dis2 = np.array(x_train_dis2)

        x_train_dis1 = self.pad(x_train_dis1)
        x_train_dis2 = self.pad(x_train_dis2)
        x_train_dis1 = x_train_dis1.reshape(len(x_train_dis1), self.config.max_len, 1)
        x_train_dis2 = x_train_dis2.reshape(len(x_train_dis2), self.config.max_len, 1)

        # 结果集one-hot，不能直接使用数字作为标签
        y_train = to_categorical(y_train)
        y_train2 = to_categorical(y_train2)

        x_valid = self.pad(x_valid)
        x_valid_dis1 = np.array(x_valid_dis1)
        x_valid_dis2 = np.array(x_valid_dis2)

        x_valid_dis1 = self.pad(x_valid_dis1)
        x_valid_dis2 = self.pad(x_valid_dis2)
        x_valid_dis1 = x_valid_dis1.reshape(len(x_valid_dis1), self.config.max_len, 1)
        x_valid_dis2 = x_valid_dis2.reshape(len(x_valid_dis2), self.config.max_len, 1)

        # 结果集one-hot，不能直接使用数字作为标签
        y_valid = to_categorical(y_valid)
        y_valid2 = to_categorical(y_valid2)

        # 初始化回调函数并用其训练
        self.callbacks = []
        self.init_callbacks_multi_dis()
        self.model.fit([x_train, x_train_dis1, x_train_dis2], [y_train, y_train2],
                       epochs=self.config.num_epochs,
                       verbose=self.config.verbose_training,
                       batch_size=self.config.batch_size,
                       # 这里随机分出一部分数据作为验证集
                       # validation_split=0.3,
                       validation_data=([x_valid, x_valid_dis1, x_valid_dis2], [y_valid, y_valid2]),
                       callbacks=self.callbacks,
                       # 平衡一下0数据的权重
                       class_weight='balanced')

    def predict(self, x, x_features=None):
        x = self.pad(x)
        if x_features is not None:
            y_pred = self.model.predict([x, x_features], batch_size=100, verbose=1)
        else:
            y_pred = self.model.predict(x, batch_size=100, verbose=1)
        return y_pred

    def predict_multi(self, x, x_features=None):
        x = self.pad(x)
        if x_features is not None:
            y_pred = self.model.predict([x, x_features], batch_size=100, verbose=1)[0]
        else:
            y_pred = self.model.predict(x, batch_size=100, verbose=1)[0]
        return y_pred

    def predict_multi_dis(self, x, x_dis1, x_dis2):
        x = self.pad(x)
        x_dis1 = self.pad(x_dis1)
        x_dis1 = x_dis1.reshape(len(x_dis1), self.config.max_len, 1)

        x_dis2 = self.pad(x_dis2)
        x_dis2 = x_dis2.reshape(len(x_dis2), self.config.max_len, 1)
        y_pred = self.model.predict([x, x_dis1, x_dis2], batch_size=100, verbose=1)[0]
        return y_pred

    def evaluate(self, model_name, y_pred, y_true):
        score_path = self.config.score_path
        fw = codecs.open(score_path, 'a', encoding='utf-8')
        y_pred = [np.argmax(y) for y in y_pred]
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        auc = categorical_metrics.multiclass_roc_auc_score(y_true, y_pred, average="weighted")
        print('\n- **Evaluation results of %s model**' % model_name)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1:', f1)
        print('AUC:', auc)
        print('Accuracy:', accuracy)
        fw.write("|%s|%.4f|%.4f|%.4f|%.4f|%.4f|\n" % (model_name, precision, recall, f1, auc, accuracy))
        fw.close()
        return precision, recall, f1, auc, accuracy



