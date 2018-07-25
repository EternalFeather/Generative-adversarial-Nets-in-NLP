# -*- coding: utf-8 -*-


class Hyperparams:
    # data
    source_train = 'corpora/train_src.txt'
    target_train = 'corpora/train_tgt.txt'
    source_test = 'corpora/test_src.txt'
    target_test = 'corpora/test_tgt.txt'
    DECODER_VOCAB = "vocabulary/de.vocab.tsv"
    ENCODER_VOCAB = "vocabulary/en.vocab.tsv"
    
    batch_size = 32
    learning_rate = 0.0001
    logdir = 'logdir'
    maxlen = 10
    min_cnt = 20
    hidden_units = 512
    num_blocks = 6
    num_epochs = 12
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False
    min_word_count = 3
