# -*- coding: utf-8 -*-


class Config(object):
    def __init__(self):
        self.level = "word"
        self.checkpoint_dir = 'modfile'
        self.logs_dir = 'logs'
        self.exp_name = None
        self.embedding_path = None
        self.embedding_path_word = None
        self.embedding_path_char = None
        self.max_len = None
        self.vocab_len = None
        self.num_epochs = 50  # 100
        self.learning_rate = 0.001
        self.optimizer = "adam"
        self.batch_size = 128
        self.verbose_training = 1
        self.checkpoint_monitor = "val_f1"
        self.checkpoint_mode = "max"
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_verbose = True
        self.early_stopping_monitor = 'val_f1'
        self.early_stopping_patience = 5
        self.early_stopping_mode = 'max'
        self.max_len_word = 60
        self.max_len_char = 300
        self.max_bag_len_word = 300
        self.max_bag_len_char = 300
        self.vocab_len_word = 307155
        self.vocab_len_char = 88
        self.char_per_word = 10
        self.embedding_path = "modfile"
        self.embedding_file = 'sst_300_dim_all.'  # 'raw_sst_300_dim_all.'
        self.result_file = './result/result_sent'
        self.bag_result_file = './result/result_bag'
        self.score_path = './result/score.md'
        self.embedding_dim = 300
        self.dropout = 0.1
        self.features_len = 0
        self.features = []
        self.classes = 35
        self.classes_multi = 2
        self.lr = 0.001
        self.sen_len = 60
        self.pre_embed = True
        self.pos_limit = 15
        self.pos_dim = 5
        self.window = 3
        self.word_dim = 300
        self.hidden_dim = 300
        self.batch_size = 128
        self.data_path = './raw_data/open_data'
        self.model_path = './modfile'
        self.mode = 'train'
        self.epochs = 200
        self.dropout = 0.5
        self.word_frequency = 5
        self.level = 'bag'

        if self.level == 'sent':
            self.bag = False
        elif self.level == 'bag':
            self.bag = True
        else:
            self.bag = True
        self.pos_num = 2 * self.pos_limit + 3
        self.relation2id = self.load_relation()
        self.num_classes = len(self.relation2id)

