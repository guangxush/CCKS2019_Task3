# -*- encoding:utf-8 -*-

from keras.engine import Input
from keras.layers import Embedding, Dropout, Conv1D, Dense, Flatten, Activation, MaxPooling1D, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from models.callbacks import categorical_metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from sklearn.metrics import precision_score, recall_score, accuracy_score
from keras.utils import to_categorical

import os
import numpy as np
import codecs


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
                mode=self.config.early_stopping_mode,
            )
        )
        self.callbacks.append(
            CSVLogger(
                filename=os.path.join(self.config.logs_dir, '%s.log' % self.config.exp_name)
            )
        )

    def load_weight(self, by_name=False):
        if self.model is None:
            raise Exception("You have to build the model first.")
        checkpoint_path = os.path.join(self.config.checkpoint_dir, '%s.hdf5' % self.config.exp_name)
        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path, by_name=by_name)
        print("Model loaded")

    # cnn基本demo
    def cnn_base(self):
        sentence = Input(shape=(self.config.max_len,), dtype='int32', name='sent_base')
        dis1 = Input(shape=(self.config.max_len,), dtype='float32', name='disinfos1')
        dis2 = Input(shape=(self.config.max_len,), dtype='float32', name='disinfos2')
        weights = np.load(os.path.join(self.config.embedding_path, self.config.embedding_file))
        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=True)
        embedding_dis1_layer = Embedding(input_dim=self.config.max_len * 2,
                                         output_dim=5,
                                         name='embedding_dis1_layer', trainable=True)

        embedding_dis2_layer = Embedding(input_dim=self.config.max_len * 2,
                                         output_dim=5,
                                         name='embedding_dis2_layer', trainable=True)

        sent_embedding = embedding_layer(sentence)
        dis1_embedding = embedding_dis1_layer(dis1)
        dis2_emdedding = embedding_dis2_layer(dis2)
        all_input = concatenate([sent_embedding, dis1_embedding, dis2_emdedding], axis=2)
        filter_length = 3
        conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='valid', strides=1, activation='relu')
        sent_c = conv_layer(all_input)
        sent_maxpooling = MaxPooling1D(pool_size=self.config.max_len - filter_length + 1)(sent_c)
        sent_conv = Flatten()(sent_maxpooling)
        sent_conv = Activation('relu')(sent_conv)
        sent = Dropout(0.5)(sent_conv)
        output = Dense(self.config.classes, activation='softmax', name='output')(sent)

        inputs = [sentence, dis1, dis2]
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=self.config.optimizer,
                           metrics=['acc'])

    def pad(self, x_data):
        return pad_sequences(x_data, maxlen=self.config.max_len, padding='post', truncating='post')

    def fit(self, x_train, x_train_dis1, x_train_dis2, y_train, x_valid, x_valid_dis1, x_valid_dis2, y_valid):
        x_train = self.pad(x_train)
        x_train_dis1 = np.array(x_train_dis1)
        x_train_dis2 = np.array(x_train_dis2)

        x_train_dis1 = self.pad(x_train_dis1)
        x_train_dis2 = self.pad(x_train_dis2)

        # 结果集one-hot，不能直接使用数字作为标签
        y_train = to_categorical(y_train)

        x_valid = self.pad(x_valid)
        x_valid_dis1 = np.array(x_valid_dis1)
        x_valid_dis2 = np.array(x_valid_dis2)

        x_valid_dis1 = self.pad(x_valid_dis1)
        x_valid_dis2 = self.pad(x_valid_dis2)

        # 结果集one-hot，不能直接使用数字作为标签
        y_valid = to_categorical(y_valid)

        # 初始化回调函数并用其训练
        self.callbacks = []
        self.init_callbacks()
        self.model.fit([x_train, x_train_dis1, x_train_dis2], y_train,
                       epochs=self.config.num_epochs,
                       verbose=self.config.verbose_training,
                       batch_size=self.config.batch_size,
                       validation_data=([x_valid, x_valid_dis1, x_valid_dis2], y_valid),
                       callbacks=self.callbacks,
                       class_weight='balanced')

    def predict(self, x, x_dis1, x_dis2):
        x = self.pad(x)
        x_dis1 = self.pad(x_dis1)
        x_dis2 = self.pad(x_dis2)
        y_pred = self.model.predict([x, x_dis1, x_dis2], batch_size=100, verbose=1)[0]
        return y_pred

    def evaluate(self, model_name, y_pred, y_true):
        score_path = self.config.score_path
        fw = codecs.open(score_path, 'a', encoding='utf-8')
        y_pred_for_f1 = y_pred
        y_true_for_f1 = to_categorical(y_true, 35)
        y_pred = [np.argmax(y) for y in y_pred]
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = self.new_f1(y_true_for_f1, y_pred_for_f1)
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

    # 根据比赛结果自定义的f1值
    def new_f1(self, all_preds, all_labels):
        n_r = int(np.sum(all_preds[:, 1:] * all_labels[:, 1:]))
        n_std = int(np.sum(all_labels[:, 1:]))
        n_sys = int(np.sum(all_preds[:, 1:]))
        try:
            precision = n_r / n_sys
            recall = n_r / n_std
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        return f1



