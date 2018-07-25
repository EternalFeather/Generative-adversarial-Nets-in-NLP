# -*- coding: utf-8 -*-
import codecs
import os
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from Transformer.transformer import Transformer
from Transformer.corpora.data_loader import Data_helper
from Transformer.config.hyperparams import Hyperparams as pm
from tqdm import tqdm
from collections import Counter


class Transformer_interface(object):
	def __init__(self):
		if os.path.exists(pm.DECODER_VOCAB):
			data_helper = Data_helper()
			# Load vocabulary
			self.de2idx, self.idx2de = data_helper.load_vocab(pm.DECODER_VOCAB)
			self.en2idx, self.idx2en = data_helper.load_vocab(pm.ENCODER_VOCAB)
		else:
			self.build_vocabulary(pm.source_train, pm.DECODER_VOCAB)
			self.build_vocabulary(pm.target_train, pm.ENCODER_VOCAB)

	def train(self):
		# Construct model
		model = Transformer()
		print("Graph loaded")
		init = tf.global_variables_initializer()

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		# Start training
		sv = tf.train.Supervisor(logdir=pm.logdir, save_model_secs=0, init_op=init)
		saver = sv.saver
		with sv.managed_session(config=config) as sess:
			for epoch in range(1, pm.num_epochs + 1):
				if sv.should_stop():
					break
				for _ in tqdm(range(model.num_batch), total=model.num_batch, ncols=70, leave=False, unit='b'):
					sess.run(model.optimizer)

				gs = sess.run(model.global_step)
				saver.save(sess, pm.logdir + '/model_epoch_{}_global_step_{}'.format(epoch, gs))

		print("MSG : Done for training!")

	def evaluate(self):
		# Load graph
		model = Transformer(trainable=False)
		print("Graph loaded")

		# Load data
		X, Sources, Targets = model.data_helper.load_test_datasets()

		# Start testing
		sv = tf.train.Supervisor()
		saver = sv.saver
		with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			saver.restore(sess, tf.train.latest_checkpoint(pm.logdir))
			print("Restored!")

			# Load Model
			mname = codecs.open(pm.logdir + '/checkpoint', 'r', encoding='utf-8').read().split('"')[1]

			# Inference
			if not os.path.exists('results'):
				os.mkdir('results')
			with codecs.open("results/" + mname, "w", encoding="utf-8") as f:
				list_of_refs, hypothesis = [], []
				num_batch = len(X) // pm.batch_size
				for i in range(num_batch):
					# Get mini batches
					x = X[i * pm.batch_size: (i + 1) * pm.batch_size]
					sources = Sources[i * pm.batch_size: (i + 1) * pm.batch_size]
					targets = Targets[i * pm.batch_size: (i + 1) * pm.batch_size]

					# Auto-regressive inference
					preds = np.zeros((pm.batch_size, pm.maxlen), dtype=np.int32)
					for j in range(pm.maxlen):
						pred = sess.run(model.predicts, {model.x: x, model.y: preds})
						preds[:, j] = pred[:, j]

					for source, target, pred in zip(sources, targets, preds):
						res = " ".join(self.idx2en[idx] for idx in pred).split("<EOS>")[0].strip()
						f.write("- source: {}\n".format(source))
						f.write("- ground truth: {}\n".format(target))
						f.write("- predict: {}\n\n".format(res))
						f.flush()

						# Bleu Score
						ref = target.split()
						predicts = res.split()
						if len(ref) > pm.min_word_count and len(predicts) > pm.min_word_count:
							list_of_refs.append([ref])
							hypothesis.append(predicts)

				score = corpus_bleu(list_of_refs, hypothesis)
				f.write("Bleu Score = {}".format(100 * score))

		print("MSG : Done for testing!")

	def build_vocabulary(self, path, fname):
		files = codecs.open(path, 'r', encoding='utf-8').read()
		words = files.split()
		wordcount = Counter(words)
		if not os.path.exists('vocabulary'):
			os.mkdir('vocabulary')
		with codecs.open(fname, 'w', encoding='utf-8') as f:
			f.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<SOS>", "<EOS>"))
			for word, count in wordcount.most_common(len(wordcount)):
				f.write("{}\t{}\n".format(word, count))


if __name__ == '__main__':
	interface = Transformer_interface()
	interface.train()
	# interface.evaluate()
