# -*- coding:utf-8 -*-
import numpy as np
import codecs
from Config.hyperparameters import Parameters as pm
from collections import Counter


class Gen_data_loader(object):
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_sentences = np.array([])
		self.sequence_batch = np.array([])
		self.num_batch = 0
		self.num_batch = 0
		self.pointer = 0

	def mini_batch(self, data_file):
		token_seqs = []
		with codecs.open(data_file, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip('\n')
				parse_line = list(map(int, line.split()))
				if pm.REAL_WORLD_DATA:
					if len(parse_line) == pm.WGAN_SEQ_LENGTH:
						token_seqs.append(parse_line)
				else:
					if len(parse_line) == pm.SEQ_LENGTH:
						token_seqs.append(parse_line)

		self.num_batch = int(len(token_seqs) / self.batch_size)
		token_seqs = token_seqs[:self.num_batch * self.batch_size]
		self.token_sentences = np.array(token_seqs)
		self.sequence_batch = np.split(self.token_sentences, self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = self.sequence_batch[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batch  # back to beginning
		return result

	def reset_pointer(self):
		self.pointer = 0


class Dis_data_loader(object):
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_sentence, self.labels = np.array([]), np.array([])
		self.gen_sentence, self.dis_sentence = np.array([]), np.array([])
		self.sentence_batches, self.labels_batches, self.gen_sentence_batches, self.dis_sentence_batches = np.array([]), np.array([]), np.array([]), np.array([])
		self.num_batch = 0
		self.pointer = 0

	def mini_batch(self, positive_file, negative_file):
		positive_example, negative_example = [], []
		with codecs.open(positive_file, 'r', encoding='utf-8') as fpo:
			for line in fpo:
				line = line.strip('\n')
				parse_line = list(map(int, line.split()))
				positive_example.append(parse_line)
		with codecs.open(negative_file, 'r', encoding='utf-8') as fne:
			for line in fne:
				line = line.strip('\n')
				parse_line = list(map(int, line.split()))
				if pm.REAL_WORLD_DATA:
					if len(parse_line) == pm.WGAN_SEQ_LENGTH:
						negative_example.append(parse_line)
				else:
					if len(parse_line) == pm.SEQ_LENGTH:
						negative_example.append(parse_line)

		self.token_sentence = np.array(positive_example + negative_example)

		# Generate labels
		positive_labels = [[0, 1] for _ in positive_example]    # one-hot vector = [negative, positive]
		negative_labels = [[1, 0] for _ in negative_example]
		self.labels = np.concatenate([positive_labels, negative_labels], axis=0)

		# Shuffle sampling
		shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
		self.token_sentence = self.token_sentence[shuffle_indices]
		self.labels = self.labels[shuffle_indices]

		# Split batches
		self.num_batch = int(len(self.labels) / self.batch_size)
		self.token_sentence = self.token_sentence[:self.num_batch * self.batch_size]
		self.labels = self.labels[:self.num_batch * self.batch_size]
		self.sentence_batches = np.split(self.token_sentence, self.num_batch, 0)
		self.labels_batches = np.split(self.labels, self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = (self.sentence_batches[self.pointer], self.labels_batches[self.pointer])
		self.pointer = (self.pointer + 1) % self.num_batch
		return result

	def reset_pointer(self):
		self.pointer = 0


class Obama_data_loader(object):
	def __init__(self, batch_size, path, data_file):
		self.batch_size = batch_size
		self.path = path
		self.data_file = data_file
		self.tokens = []
		self.sentence_batch = np.array([])
		self.token_batch = np.array([])
		self.num_batch = 0
		self.pointer = 0

	def build_vocabulary(self):
		files = codecs.open(self.data_file, 'r', encoding='utf-8').read()
		words = files.split()
		wordcount = Counter(words)
		with codecs.open(self.path, 'w', encoding='utf-8') as f:
			f.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<SOS>", "<EOS>"))
			for word, count in wordcount.most_common(9996):
				f.write("{}\t{}\n".format(word, count))

	def load_dataset(self):
		sentences = [line for line in codecs.open(self.data_file, 'r', encoding='utf-8').read().split('\n') if line]
		word2idx, idx2word = self.load_vocabulary()

		token_list, sources = [], []
		for source in sentences:
			if len(source.split()) >= 19:
				x = [word2idx.get(word, 1) for word in (" ".join(source.split()[:19]) + " <EOS>").split()]
			else:
				# x = [word2idx.get(word, 1) for word in (source + (19 - len(source.split())) * " <PAD>" + " <EOS>").split()]
				continue
			token_list.append(x)
			sources.append(source)
		return token_list, sources

	def load_vocabulary(self):
		vocab = [line.split()[0] for line in codecs.open(self.path, 'r', encoding='utf-8').read().splitlines()]
		word2idx = {word: idx for idx, word in enumerate(vocab)}
		idx2word = {word2idx[word]: word for word in word2idx}
		return word2idx, idx2word

	def mini_batch(self):
		self.tokens, sentences = self.load_dataset()
		self.num_batch = int(len(sentences) / self.batch_size)
		sentences = sentences[:self.num_batch * self.batch_size]
		tokens = self.tokens[:self.num_batch * self.batch_size]
		self.sentence_batch = np.split(np.array(sentences), self.num_batch, 0)
		self.token_batch = np.split(np.array(tokens), self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = self.token_batch[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batch
		return result

	def reset_pointer(self):
		self.pointer = 0


class Chinese_qtans_data_loader(object):
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.token_seqs = []
		self.sentences = np.array([])
		self.tokens = np.array([])
		self.sequence_batch = np.array([])
		self.sentence_batch = np.array([])
		self.num_batch = 0
		self.pointer = 0

	def build_vocabulary(self, path, datafile):
		files = codecs.open(datafile, 'r', encoding='utf-8').read()
		words = files.split()
		wordcount = Counter(words)
		with codecs.open(path, 'w', encoding='utf-8') as f:
			f.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<SOS>", "<EOS>"))
			for word, count in wordcount.most_common(len(wordcount)):
				f.write("{}\t{}\n".format(word, count))

	def load_dataset(self, path, datafile):
		sentences = [line for line in codecs.open(datafile, 'r', encoding='utf-8').read().split('\n') if line]
		word2idx, idx2word = self.load_vocabulary(path)

		token_list, sources = [], []
		for source in sentences:
			x = [word2idx.get(word, 1) for word in (source + " <EOS>").split()]
			token_list.append(x)
			sources.append(source)
		return token_list, sources

	def load_vocabulary(self, path):
		vocab = [line.split()[0] for line in codecs.open(path, 'r', encoding='utf-8').read().splitlines()]
		word2idx = {word: idx for idx, word in enumerate(vocab)}
		idx2word = {word2idx[word]: word for word in word2idx}
		return word2idx, idx2word

	def mini_batch(self, path, datafile):
		self.token_seqs, sentences = self.load_dataset(path, datafile)
		self.num_batch = int(len(sentences) / self.batch_size)
		sentences = sentences[:self.num_batch * self.batch_size]
		tokens = self.token_seqs[:self.num_batch * self.batch_size]
		self.sentences, self.tokens = np.array(sentences), np.array(tokens)
		self.sequence_batch = np.split(self.tokens, self.num_batch, 0)
		self.sentence_batch = np.split(self.sentences, self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = self.sequence_batch[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batch
		return result

	def reset_pointer(self):
		self.pointer = 0


class WGAN_data_loader(object):
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.lines_batch = np.array([])
		self.tokens_batch = np.array([])
		self.num_batch = 0
		self.vocab_size = 0
		self.pointer = 0

	def build_vocabulary(self, path, datafile, vocab_size, char=True):
		files = codecs.open(datafile, 'r', encoding='utf-8').read()
		if char:
			words = []
			files = files.split('\n')
			for word in files:
				word = tuple(word)
				words.append(word)
		else:
			words = files.split()
		wordcount = Counter(c for line in words for c in line if c != ' ')
		with codecs.open(path, 'w', encoding='utf-8') as f:
			f.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<SOS>", "<EOS>", "<SPA>"))
			for word, count in wordcount.most_common(len(wordcount)-5):
				f.write("{}\t{}\n".format(word, count))
		self.vocab_size = len(wordcount) - 5

	def load_dataset(self, datafile, path, seq_length):
		sentences = [line for line in codecs.open(datafile, 'r', encoding='utf-8').read().split('\n') if line]
		word2idx, idx2word = self.load_vocabulary(path)

		token_list, sources = [], []
		for source in sentences:
			temp = []
			for char in source:
				temp.append(char)

			if len(temp) >= seq_length:
				temp = temp[: seq_length - 1]
				source = source[: seq_length - 1]

			x = [word2idx.get(word, 1) for word in (temp + ["<EOS>"])]
			if len(x) < seq_length:
				x += ["0"] * (seq_length - len(x))

			token_list.append(x)
			sources.append(source)
		return token_list, sources

	def load_vocabulary(self, path):
		vocab = [line.split()[0] for line in codecs.open(path, 'r', encoding='utf-8').read().splitlines()]
		vocab[4] = " "
		word2idx = {word: idx for idx, word in enumerate(vocab)}
		idx2word = {word2idx[word]: word for word in word2idx}
		return word2idx, idx2word

	def mini_batch(self, path, datafile, seq_length):
		token_seqs, sentences = self.load_dataset(path, datafile, seq_length)
		self.num_batch = int(len(sentences) / self.batch_size)
		sentences = sentences[:self.batch_size * self.num_batch]
		tokens = token_seqs[:self.batch_size * self.num_batch]
		sentences, tokens = np.array(sentences), np.array(tokens)
		self.lines_batch = np.split(sentences, self.num_batch, 0)
		self.tokens_batch = np.split(tokens, self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = self.tokens_batch[self.pointer]
		self.pointer = (self.pointer + 1) % self.num_batch
		return result

	def reset_pointer(self):
		self.pointer = 0


class WGAN_disc_data_loader(object):
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.positive_examples_batch = np.array([])
		self.negative_examples_batch = np.array([])
		self.positive_labels_batch = np.array([])
		self.negative_labels_batch = np.array([])
		self.num_batch = 0
		self.pointer = 0

	def mini_batch(self, positive_file, negative_file):
		positive_examples, negative_examples = [], []
		with codecs.open(positive_file, 'r', encoding='utf-8') as fpo:
			for line in fpo:
				line = line.strip('\n')
				parse_line = list(map(int, line.split()))
				positive_examples.append(parse_line)
		with codecs.open(negative_file, 'r', encoding='utf-8') as fng:
			for line in fng:
				line = line.strip('\n')
				parse_line = list(map(int, line.split()))
				negative_examples.append(parse_line)

		positive_labels = [[1] for _ in positive_examples]
		negative_labels = [[0] for _ in negative_examples]

		# Shuffle sampling
		shuffle_indices = np.random.permutation(np.arange(len(positive_labels)-16))
		positive_examples = np.array(positive_examples)[shuffle_indices]
		negative_examples = np.array(negative_examples)[shuffle_indices]
		positive_labels = np.array(positive_labels)[shuffle_indices]
		negative_labels = np.array(negative_labels)[shuffle_indices]

		# Split
		self.num_batch = int(len(positive_labels) / self.batch_size)
		positive_examples = positive_examples[:self.batch_size * self.num_batch]
		negative_examples = negative_examples[:self.batch_size * self.num_batch]
		positive_labels = positive_labels[:self.batch_size * self.num_batch]
		negative_labels = negative_labels[:self.batch_size * self.num_batch]
		self.positive_examples_batch = np.split(positive_examples, self.num_batch, 0)
		self.negative_examples_batch = np.split(negative_examples, self.num_batch, 0)
		self.positive_labels_batch = np.split(positive_labels, self.num_batch, 0)
		self.negative_labels_batch = np.split(negative_labels, self.num_batch, 0)
		self.reset_pointer()

	def next_batch(self):
		result = (self.positive_examples_batch[self.pointer], self.positive_labels_batch[self.pointer], self.negative_examples_batch[self.pointer], self.negative_labels_batch[self.pointer])
		self.pointer = (self.pointer + 1) % self.num_batch
		return result

	def reset_pointer(self):
		self.pointer = 0
