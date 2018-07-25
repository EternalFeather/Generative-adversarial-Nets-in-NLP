# -*- coding:utf-8 -*-


class Parameters(object):

	# Adversarial
	RANDOM_SEED = 88
	START_TOKEN = 0
	BATCH_SIZE = 64
	VOCAB_SIZE = 5000
	# VOCAB_SIZE = 4492
	# VOCAB_SIZE = 10000
	GENERATED_NUM = 10000
	ATTENTION_PRE_TRAIN_EPOCH = 300
	K = 3
	TOTAL_BATCHES = 120
	N_GRAM = 2
	ATT_LEARNING_RATE = 0.004
	MONTE_CARLO_TURNS = 16
	ADVERSARIAL_DROPOUT = 0.75
	DECAY_STEPS = 20
	STARICASE = True
	CLIP_WEIGHT = False
	BiLSTM = False
	DYNAMIC_LR = False
	TEACHER_FORCING = False
	MODEL_PATH = "Log/target_params_py3.pkl"
	REAL_DATA_PATH = "Datasets/Oracle/Real_datasets.txt"
	REAL_DIS_DATA_PATH = "Datasets/Oracle/Real_dis_datasets.txt"
	PRE_GENERATOR_DATA = "Datasets/Oracle/Pre_train_generator_datasets.txt"
	G_NEG_SAMPLING_DATA = "Datasets/Oracle/Generator_negative_sampling_datasets.txt"
	ADVERSARIAL_G_DATA = "Datasets/Oracle/Adversarial_generator_sampling_datasets.txt"
	ADVERSARIAL_NEG_DATA = "Datasets/Oracle/Adversarial_negative_datasets.txt"

	CN_REAL_DATA_PATH = "Datasets/Chinese_quatrains/Real_datasets.txt"
	CN_PRE_GENERATOR_DATA = "Datasets/Chinese_quatrains/Pre_train_generator_datasets.txt"
	CHINESE_QUATRAINS_FIVE = "Datasets/Chinese_quatrains/Chinese_quatrains_5.txt"
	CN_G_NEG_SAMPLING_DATA = "Datasets/Chinese_quatrains/Generator_negative_sampling_datasets.txt"
	CN_ADVERSARIAL_G_DATA = "Datasets/Chinese_quatrains/Adversarial_generator_sampling_datasets.txt"
	CN_ADVERSARIAL_NEG_DATA = "Datasets/Chinese_quatrains/Adversarial_negative_datasets.txt"

	CHINESE_VOCAB_FIVE = "Datasets/Chinese_quatrains/Chinese_quatrains_5.vocab"
	CHINESE_5_TRANSLATE = "Datasets/Chinese_quatrains/Chinese_translate_5.txt"

# --------

	OB_REAL_DATA_PATH = "Datasets/Obama/Real_datasets.txt"
	OB_PRE_GENERATOR_DATA = "Datasets/Obama/Pre_train_generator_datasets.txt"
	OB_SPEECH = "Datasets/Obama/Obama_speech.txt"
	OB_G_NEG_SAMPLING_DATA = "Datasets/Obama/Generator_negative_sampling_datasets.txt"
	OB_ADVERSARIAL_G_DATA = "Datasets/Obama/Adversarial_generator_sampling_datasets.txt"
	OB_ADVERSARIAL_NEG_DATA = "Datasets/Obama/Adversarial_negative_datasets.txt"

	OB_VOCAB = "Datasets/Obama/Obama_speech.vocab"
	OB_TRANSLATE = "Datasets/Obama/Obama_translate.txt"

	REAL_WORLD_DATA = False
	WGAN_VOCAB_SIZE = 125
	WGAN_SEQ_LENGTH = 30
	VOCAB_PATH = "Datasets/Google_Billion_Corpus/vocab_file"
	REAL_DATA = "Datasets/Google_Billion_Corpus/real_data.txt"
	DATASET = "Datasets/Google_Billion_Corpus/google_billion_corpus.txt"
	FAKE_DATA = "Datasets/Google_Billion_Corpus/fake_datasets/fake_data"
	NEG_DATA = "Datasets/Google_Billion_Corpus/negative_data.txt"

	# Generator
	EMB_SIZE = 32
	HIDDEN_SIZE = 32
	SEQ_LENGTH = 20
	# SEQ_LENGTH = 6
	LEARNING_RATE = 0.1
	REWARD_GAMMA = 0.95
	G_PRE_TRAIN_EPOCH = 250
	UPDATE_RATE = 0.8
	G_STEP = 1

	# Discriminator
	DIS_EMB_SIZE = 64
	NUM_CLASSES = 2
	FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
	# FILTER_SIZES = [1, 2, 3]
	NUM_FILTERS = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
	# NUM_FILTERS = [100, 200, 160]
	D_LEARNING_RATE = 1e-4
	L2_REG_LAMBDA = 0.2
	D_PRE_TRAIN_EPOCH = 50
	# D_PRE_TRAIN_EPOCH = 30
	D_DROP_KEEP_PROB = 0.75
	D_STEP = 5
