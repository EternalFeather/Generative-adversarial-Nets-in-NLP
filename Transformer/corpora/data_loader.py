# -*- coding: utf-8 -*-
import codecs
import numpy as np
import tensorflow as tf
from Transformer.config.hyperparams import Hyperparams as pm


class Data_helper(object):
    def __init__(self):
        self.pointer = 0

    def mini_batch(self):
        X, Y = self.load_train_datasets()
        num_batch = len(X) // pm.batch_size
        X = tf.convert_to_tensor(X, tf.int32)
        Y = tf.convert_to_tensor(Y, tf.int32)

        # Input Queue by CPU
        input_queues = tf.train.slice_input_producer([X, Y])

        # Get mini batch from Queue
        x, y = tf.train.shuffle_batch(input_queues,
                                    num_threads=8,
                                    batch_size=pm.batch_size,
                                    capacity=pm.batch_size * 64,    # Max_number of batches in queue
                                    min_after_dequeue=pm.batch_size * 32,  # Min_number of batches in queue after dequeue
                                    allow_smaller_final_batch=False)

        return x, y, num_batch

    def load_train_datasets(self):
        de_sents = [line for line in codecs.open(pm.source_train, 'r', 'utf-8').read().split("\n") if line]
        en_sents = [line for line in codecs.open(pm.target_train, 'r', 'utf-8').read().split("\n") if line]
        x, y, sources, targets = self.generate(de_sents, en_sents)

        return x, y

    def load_test_datasets(self):
        de_sents = [line for line in codecs.open(pm.source_test, 'r', 'utf-8').read().split("\n") if line]
        en_sents = [line for line in codecs.open(pm.target_test, 'r', 'utf-8').read().split("\n") if line]
        x, y, sources, targets = self.generate(de_sents, en_sents)

        return x, sources, targets

    def generate(self, source_sents, target_sents):
        de2idx, idx2de = self.load_vocab(pm.DECODER_VOCAB)
        en2idx, idx2en = self.load_vocab(pm.ENCODER_VOCAB)

        x_list, y_list, Sources, Targets = [], [], [], []
        for source_sent, target_sent in zip(source_sents, target_sents):
            x = [de2idx.get(word, 1) for word in (source_sent + " <EOS>").split()]
            y = [en2idx.get(word, 1) for word in (target_sent + " <EOS>").split()]
            if max(len(x), len(y)) <= pm.maxlen:
                x_list.append(np.array(x))
                y_list.append(np.array(y))
                Sources.append(source_sent)
                Targets.append(target_sent)

        # Padding 0(<PAD>)
        x_np = np.zeros([len(x_list), pm.maxlen], np.int32)
        y_np = np.zeros([len(y_list), pm.maxlen], np.int32)
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            x_np[i] = np.lib.pad(x, [0, pm.maxlen - len(x)], 'constant', constant_values=(0, 0))
            y_np[i] = np.lib.pad(y, [0, pm.maxlen - len(y)], 'constant', constant_values=(0, 0))

        return x_np, y_np, Sources, Targets

    def load_vocab(self, file):
        vocab = [line.split()[0] for line in codecs.open(file, 'r', encoding='utf-8').read().splitlines() if int(line.split()[1]) >= pm.min_cnt]
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {word2idx[word]: word for word in word2idx}
        return word2idx, idx2word

    def next(self, X, Sources, Targets, num_batch):
        x = X[self.pointer * pm.batch_size: (self.pointer + 1) * pm.batch_size]
        sources = Sources[self.pointer * pm.batch_size: (self.pointer + 1) * pm.batch_size]
        targets = Targets[self.pointer * pm.batch_size: (self.pointer + 1) * pm.batch_size]
        self.pointer = (self.pointer + 1) % num_batch

        return x, sources, targets

    def reset_pointer(self):
        self.pointer = 0
