# -*- encoding:utf-8 -*-

from keras.engine import Input
from keras.layers import Embedding, Dropout, Conv1D, Conv2D, MaxPooling2D, Lambda, LSTM, CuDNNLSTM, Dense, concatenate, \
    TimeDistributed, Bidirectional, GlobalMaxPool1D, GlobalAvgPool1D, Reshape, Flatten, Activation, BatchNormalization, \
    Dot, Permute, Multiply, Add, RepeatVector, GlobalMaxPooling1D, MaxPooling1D, ZeroPadding1D
from keras.activations import softmax
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical
from models.callbacks import distance_metrics, categorical_metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from models.cnn import AttConv1D

import os, codecs
import numpy as np


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

    def init_callbacks(self, distance=False, feature=False, co_ae=False, align=False, adv=False):
        if distance:
            self.callbacks.append(distance_metrics)
        else:
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

    def load_weight(self, by_name=False):
        if self.model is None:
            raise Exception("You have to build the model first.")
        checkpoint_path = os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name)
        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path, by_name=by_name)
        print("Model loaded")

    def siamese_cnn(self, distance=False, manhattan=False):
        weights = np.load(
            os.path.join(self.config.embedding_path, self.config.level + '_level', self.config.embedding_file))
        sentence1 = Input(shape=(self.config.max_len,), dtype='int32', name='sent1_base')
        sentence2 = Input(shape=(self.config.max_len,), dtype='int32', name='sent2_base')
        features = Input(shape=(self.config.features_len,), dtype='float32', name='features')
        embedding_layer = Embedding(input_dim=self.config.vocab_len + 2,
                                    output_dim=self.config.embedding_dim,
                                    weights=[weights], name='embedding_layer', trainable=True)
        sent1_embedding = embedding_layer(sentence1)
        sent2_embedding = embedding_layer(sentence2)

        filter_lengths = [2, 3, 4, 5]
        sent1_conv_layers = []
        sent2_conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = Conv1D(filters=200, kernel_size=filter_length, padding='valid',
                                activation='relu', strides=1)
            sent1_c = conv_layer(sent1_embedding)
            sent2_c = conv_layer(sent2_embedding)
            sent1_maxpooling = MaxPooling1D(pool_size=self.config.max_len - filter_length + 1)(sent1_c)
            sent1_flatten = Flatten()(sent1_maxpooling)
            sent1_conv_layers.append(sent1_flatten)
            sent2_maxpooling = MaxPooling1D(pool_size=self.config.max_len - filter_length + 1)(sent2_c)
            sent2_flatten = Flatten()(sent2_maxpooling)
            sent2_conv_layers.append(sent2_flatten)
        sent1_conv = concatenate(inputs=sent1_conv_layers)
        sent2_conv = concatenate(inputs=sent2_conv_layers)

        if distance:
            distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([sent1_conv, sent2_conv])
            self.model = Model([sentence1, sentence2], distance)
            self.model.compile(loss=contrastive_loss, optimizer=self.config.optimizer, metrics=[acc])
        elif manhattan:
            distance = Lambda(manhattan_distance, output_shape=eucl_dist_output_shape)([sent1_conv, sent2_conv])
            self.model = Model([sentence1, sentence2], distance)
            self.model.compile(loss='binary_crossentropy',
                               optimizer=self.config.optimizer,
                               metrics=['acc'])
        else:
            sent = concatenate([sent1_conv, sent2_conv])
            sent = BatchNormalization(name='sent_representation')(sent)
            if self.config.features_len > 0:
                sent = concatenate([sent, features], axis=-1)
            x = Dense(400, activation='relu', name='dense1')(sent)
            x = BatchNormalization(name='batch_normal2')(x)
            x = Dense(400, activation='relu', name='dense2')(x)
            x = BatchNormalization(name='batch_normal3')(x)
            # simi = Dense(3, activation='softmax')(x)
            simi = Dense(1, activation='sigmoid', name='simi')(x)

            if self.config.features_len == 0:
                self.model = Model(inputs=[sentence1, sentence2], outputs=simi)
            else:
                self.model = Model(inputs=[sentence1, sentence2, features], outputs=simi)
            self.model.compile(loss='binary_crossentropy', optimizer=self.config.optimizer,
                               metrics=['binary_accuracy'])

    def siamese_att_cnn(self, distance=False, manhattan=False):
        weights = np.load(
            os.path.join(self.config.embedding_path, self.config.level + '_level', self.config.embedding_file))
        sentence1 = Input(shape=(self.config.max_len,), dtype='int32', name='sent1_base')
        sentence2 = Input(shape=(self.config.max_len,), dtype='int32', name='sent2_base')
        features = Input(shape=(self.config.features_len,), dtype='float32', name='features')
        embedding_layer = Embedding(input_dim=self.config.vocab_len + 2,
                                    output_dim=self.config.embedding_dim,
                                    weights=[weights], name='embedding_layer', trainable=True)
        sent1_embedding = embedding_layer(sentence1)
        sent2_embedding = embedding_layer(sentence2)

        attention = Dot(axes=-1)([sent1_embedding, sent2_embedding])
        wb = Lambda(lambda x: softmax(x, axis=1), output_shape=lambda x: x)(attention)
        wa = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2), output_shape=lambda x: x)(attention))
        sent1_ = Dot(axes=1)([wa, sent2_embedding])
        sent2_ = Dot(axes=1)([wb, sent1_embedding])
        neg = Lambda(lambda x: -x, output_shape=lambda x: x)
        substract1 = Add()([sent1_embedding, neg(sent1_)])
        mutiply1 = Multiply()([sent1_embedding, sent1_])
        substract2 = Add()([sent2_embedding, neg(sent2_)])
        mutiply2 = Multiply()([sent2_embedding, sent2_])

        sent1_att = concatenate([sent1_, substract1, mutiply1])
        sent2_att = concatenate([sent2_, substract2, mutiply2])

        filter_lengths = [2, 3, 4, 5]
        sent1_conv_layers = []
        sent2_conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = AttConv1D(filters=200, kernel_size=filter_length, padding='same',
                                   activation='relu', strides=1)
            sent1_conv = conv_layer([sent1_embedding, sent1_att])
            sent2_conv = conv_layer([sent2_embedding, sent2_att])
            sent1_conv = MaxPooling1D(pool_size=self.config.max_len)(sent1_conv)
            flatten = Flatten()(sent1_conv)
            sent1_conv_layers.append(flatten)
            sent2_conv = MaxPooling1D(pool_size=self.config.max_len)(sent2_conv)
            flatten = Flatten()(sent2_conv)
            sent2_conv_layers.append(flatten)
        sent1_conv = concatenate(inputs=sent1_conv_layers)
        sent2_conv = concatenate(inputs=sent2_conv_layers)

        if distance:
            distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([sent1_conv, sent2_conv])
            self.model = Model([sentence1, sentence2], distance)
            self.model.compile(loss=contrastive_loss, optimizer=self.config.optimizer, metrics=[acc])
        elif manhattan:
            distance = Lambda(manhattan_distance, output_shape=eucl_dist_output_shape)([sent1_conv, sent2_conv])
            self.model = Model([sentence1, sentence2], distance)
            self.model.compile(loss='binary_crossentropy',
                               optimizer=self.config.optimizer,
                               metrics=['acc'])
        else:
            sent = concatenate([sent1_conv, sent2_conv])
            sent = BatchNormalization(name='sent_representation')(sent)
            if self.config.features_len > 0:
                sent = concatenate([sent, features], axis=-1)
            x = Dense(400, activation='relu', name='dense1')(sent)
            x = BatchNormalization(name='batch_normal2')(x)
            x = Dense(400, activation='relu', name='dense2')(x)
            x = BatchNormalization(name='batch_normal3')(x)
            # simi = Dense(3, activation='softmax')(x)
            simi = Dense(1, activation='sigmoid', name='simi')(x)

            if self.config.features_len == 0:
                self.model = Model(inputs=[sentence1, sentence2], outputs=simi)
            else:
                self.model = Model(inputs=[sentence1, sentence2, features], outputs=simi)
            self.model.compile(loss='binary_crossentropy', optimizer=self.config.optimizer,
                               metrics=['binary_accuracy'])

    def pad(self, x_data_a, x_data_b):
        return pad_sequences(x_data_a, maxlen=self.config.max_len, padding='post', truncating='post'), \
               pad_sequences(x_data_b, maxlen=self.config.max_len, padding='post', truncating='post')

    def fit(self, x_train_a, x_train_b, y_train, x_valid_a, x_valid_b, y_valid, train_features=None,
            valid_features=None, distance=False):
        x_train_a, x_train_b = self.pad(x_train_a, x_train_b)
        x_valid_a, x_valid_b = self.pad(x_valid_a, x_valid_b)
        y_train = np.asarray(y_train)
        y_valid = np.asarray(y_valid)
        self.callbacks = []
        if train_features is not None:
            train_features = np.asarray(train_features)
            valid_features = np.asarray(valid_features)
            self.init_callbacks(feature=True)
            self.model.fit([x_train_a, x_train_b, train_features], y_train,
                           epochs=self.config.num_epochs,
                           verbose=self.config.verbose_training,
                           batch_size=self.config.batch_size,
                           validation_data=([x_valid_a, x_valid_b, valid_features], y_valid),
                           callbacks=self.callbacks)
        else:
            if distance:
                self.config.checkpoint_monitor = "val_acc"
                self.config.early_stopping_monitor = 'val_acc'
            self.init_callbacks(distance=distance)
            self.model.fit([x_train_a, x_train_b], y_train,
                           epochs=self.config.num_epochs,
                           verbose=self.config.verbose_training,
                           batch_size=self.config.batch_size,
                           validation_data=([x_valid_a, x_valid_b], y_valid),
                           callbacks=self.callbacks)

    def predict(self, x_a, x_b, x_features=None):
        x_a, x_b = self.pad(x_a, x_b)
        if x_features is not None:
            y_pred = self.model.predict([x_a, x_b, x_features], batch_size=100, verbose=1)
        else:
            y_pred = self.model.predict([x_a, x_b], batch_size=100, verbose=1)
        return y_pred

    def evaluate(self, y_pred, y_true, distance=False):
        if y_pred.shape[-1] > 1:
            y_pred = [np.argmax(y) for y in y_pred]
        else:
            if distance:
                y_pred = [y[0] < 0.5 for y in y_pred]
            else:
                y_pred = [y[0] > 0.5 for y in y_pred]
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        # auc = roc_auc_score(y_true, y_pred, average='micro')
        print('\n- **Evaluation results of %s Categorical mixed model**' % self.config.exp_name)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1:', f1)
        print('Accuracy:', accuracy)
        # print 'Auc:', auc
        return precision, recall, f1, accuracy  # , auc


