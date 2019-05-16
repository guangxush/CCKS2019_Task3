# -*- coding: utf-8 -*-
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np


class CategoricalMetrics(Callback):
    def __init__(self):
        super(CategoricalMetrics, self).__init__()

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_aucs = []

    def on_epoch_end(self, epoch, logs={}):
        valid_results = self.model.predict([self.validation_data[0]])
        valid_y_pred = valid_results
        # if isinstance(valid_results, list):
        #     valid_results = valid_results[-1]
        # if valid_results.shape[-1] > 1:
        #     valid_y_pred = [np.argmax(valid_result) for valid_result in valid_results]
        # else:
        #     valid_y_pred = [valid_result[0] > 0.5 for valid_result in valid_results]
        valid_y_true = self.validation_data[1]
        # if valid_y_true.shape[-1] > 1:
        #     valid_y_true = [np.argmax(y) for y in valid_y_true]
        # _val_f1 = f1_score(valid_y_true, valid_y_pred)
        # 替换成比赛中需要的f1
        _val_f1 = self.new_f1(valid_y_true, valid_y_pred)
        _val_recall = recall_score(valid_y_true, valid_y_pred)
        _val_precision = precision_score(valid_y_true, valid_y_pred)
        _val_auc = roc_auc_score(valid_y_true, valid_y_pred)
        logs['val_precisions'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1
        logs['val_auc'] = _val_auc
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_aucs.append(_val_auc)
        print('- val_precision: %.4f - val_recall: %.4f  - val_f1: %.4f - val_auc: %.4f' %
              (_val_precision, _val_recall, _val_f1, _val_auc))
        return

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


class CategoricalFeatureMetrics(Callback):
    def __init__(self):
        super(CategoricalFeatureMetrics, self).__init__()

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_aucs = []

    def on_epoch_end(self, epoch, logs={}):
        valid_results = self.model.predict([self.validation_data[0], self.validation_data[1], self.validation_data[2]])
        valid_y_pred = [valid_result[0] > 0.5 for valid_result in valid_results]
        valid_y_true = self.validation_data[3]
        _val_f1 = f1_score(valid_y_true, valid_y_pred)
        _val_recall = recall_score(valid_y_true, valid_y_pred)
        _val_precision = precision_score(valid_y_true, valid_y_pred)
        _val_auc = roc_auc_score(valid_y_true, valid_y_pred)
        logs['val_precisions'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1
        logs['val_auc'] = _val_auc
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_aucs.append(_val_auc)
        print('- val_precision: %.4f - val_recall: %.4f  - val_f1: %.4f - val_auc: %.4f' %
              (_val_precision, _val_recall, _val_f1, _val_auc))
        return


class DistanceMetrics(Callback):
    def __init__(self):
        super(DistanceMetrics, self).__init__()

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_aucs = []

    def on_epoch_end(self, epoch, logs={}):
        valid_results = self.model.predict([self.validation_data[0], self.validation_data[1]])
        if isinstance(valid_results, list):
            valid_results = valid_results[-1]
        valid_y_pred = [valid_result[0] < 0.5 for valid_result in valid_results]
        valid_y_true = self.validation_data[2]
        _val_f1 = f1_score(valid_y_true, valid_y_pred)
        _val_recall = recall_score(valid_y_true, valid_y_pred)
        _val_precision = precision_score(valid_y_true, valid_y_pred)
        _val_auc = roc_auc_score(valid_y_true, valid_y_pred)
        logs['val_precisions'] = _val_precision
        logs['val_recall'] = _val_recall
        logs['val_f1'] = _val_f1
        logs['val_auc'] = _val_auc
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_aucs.append(_val_auc)
        print('- val_precision: %.4f - val_recall: %.4f  - val_f1: %.4f - val_auc: %.4f' %
              (_val_precision, _val_recall, _val_f1, _val_auc))
        return


distance_metrics = DistanceMetrics()

categorical_metrics = CategoricalMetrics()



