# -*- coding:utf-8 -*-
import random
from Config.hyperparameters import Parameters as pm
import numpy as np
import tensorflow as tf
from Datasets.dataloader import Gen_data_loader, Dis_data_loader, Chinese_qtans_data_loader
import pickle
from Model.corpus_lstm import Corpus_lstm
import codecs, os
import matplotlib.pyplot as plt
from Model.discriminator import Discriminator
from Model.attention_reward import Attention_reward
from time import time


class SeqGAN(object):
	def __init__(self):
		random.seed(pm.RANDOM_SEED)
		np.random.seed(pm.RANDOM_SEED)
		assert pm.START_TOKEN == 0

# Initialize models ------------------

		# Init
		gen_data_loader = Gen_data_loader(pm.BATCH_SIZE)
		likelihood_data_loader = Gen_data_loader(pm.BATCH_SIZE)     # For Testing
		dis_data_loader = Dis_data_loader(pm.BATCH_SIZE)
		# chinese_quatrains_data_loader = Chinese_qtans_data_loader(pm.BATCH_SIZE)
		discriminator = Discriminator(pm.SEQ_LENGTH, pm.NUM_CLASSES, pm.VOCAB_SIZE, pm.DIS_EMB_SIZE, pm.FILTER_SIZES, pm.NUM_FILTERS, pm.D_LEARNING_RATE, pm.L2_REG_LAMBDA)
		target_params = pickle.load(open(pm.MODEL_PATH, 'rb'), encoding='latin1')   # Oracle LSTM_model for corpus generation
		corpus_lstm = Corpus_lstm(pm.VOCAB_SIZE, pm.BATCH_SIZE, pm.EMB_SIZE, pm.HIDDEN_SIZE, pm.SEQ_LENGTH, pm.START_TOKEN, target_params)
		attention_mechanism = Attention_reward(pm.VOCAB_SIZE, pm.BATCH_SIZE, pm.EMB_SIZE, pm.HIDDEN_SIZE, pm.SEQ_LENGTH, pm.START_TOKEN, pm.ATT_LEARNING_RATE, pm.DECAY_STEPS, True)

# End initialize models --------------

# Config tensorflow session ----------

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())

# End session configuration ----------

# Pre-train Generator with real datasets from corpus_lstm model ----------------

		if not os.path.exists('Datasets/Oracle'):
			os.mkdir('Datasets/Oracle')

		# Convert sentences to token_ids

		# chinese_quatrains_data_loader.build_vocabulary(pm.CHINESE_VOCAB_FIVE, pm.CHINESE_QUATRAINS_FIVE)
		# chinese_quatrains_data_loader.mini_batch(pm.CHINESE_VOCAB_FIVE, pm.CHINESE_QUATRAINS_FIVE)
		# with codecs.open(pm.REAL_DATA_PATH, 'w', encoding='utf-8') as f:
		# 	for data in chinese_quatrains_data_loader.token_seqs:
		# 		buffer = " ".join(str(i) for i in data)
		# 		f.write(buffer + '\n')

		# Generate 1W sequences of length 20 as the training set S for the generator model
		self.generate_samples(sess, corpus_lstm, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.REAL_DATA_PATH, pm.REAL_DIS_DATA_PATH)
		gen_data_loader.mini_batch(pm.REAL_DATA_PATH)

		log = codecs.open("Log/experiment-log.txt", 'w', encoding='utf-8')

# End of pre-train Generator ---------------

# Pre-train Attention -------------------------------

		# Pre-train Attention_Mechanism
		gen = []
		print("MSG : Start Pre-train Attention_Mechanism...")
		log.write("Pre-train Attention_Mechanism...\n")
		log.flush()
		for epoch in range(pm.ATTENTION_PRE_TRAIN_EPOCH):
			pretrain_loss = self.att_pre_train_loss(sess, attention_mechanism, gen_data_loader)
			if epoch % 5 == 0:
				self.att_generate_samples(sess, attention_mechanism, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.PRE_GENERATOR_DATA, gen_data_loader)
				likelihood_data_loader.mini_batch(pm.PRE_GENERATOR_DATA)
				test_loss = self.target_loss(sess, corpus_lstm, likelihood_data_loader)
				gen.append(test_loss)
				print("Pre-train Attention Epoch: {}, Test_loss(NLL): {}, Pretrain_loss: {}".format(epoch, test_loss, pretrain_loss))
				buffer = "Pre-train Generator Epoch:\t{}\tNLL:\t{}\tAttention_Loss:\t{}\n".format(str(epoch), str(test_loss), str(pretrain_loss))
				log.write(buffer)
				log.flush()

		pretrain_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
		self.matplotformat(ax1, gen, 'Pre-train Generator', pm.ATTENTION_PRE_TRAIN_EPOCH)

# End of Pre-train Attention ------------------------

# Pre-train Discriminator with positive datasets(real) and negative datasets from Generator model ---------------

		# Pre-train Discriminator
		dis = []
		temp_min, temp_max = 10000.0, -10000.0
		print("MSG : Start Pre-train Discriminator...")
		log.write("Pre-train Discriminator...\n")
		log.flush()
		for epoch in range(pm.D_PRE_TRAIN_EPOCH):
			# self.generate_samples(sess, generator, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.G_NEG_SAMPLING_DATA)
			self.att_generate_samples(sess, attention_mechanism, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.G_NEG_SAMPLING_DATA, gen_data_loader)
			# self.dis_generate_samples(sess, attention_mechanism, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.G_NEG_SAMPLING_DATA, gen_data_loader)
			dis_data_loader.mini_batch(pm.REAL_DATA_PATH, pm.G_NEG_SAMPLING_DATA)
			for _ in range(pm.K):
				test_loss = self.dis_pre_train_loss(sess, discriminator, dis_data_loader)
				if test_loss < temp_min:
					temp_min = test_loss
				if test_loss > temp_max:
					temp_max = test_loss

			if epoch % 5 == 0:
				dis.append(test_loss)
				print("Pre-train Dis Epoch: {}, Test_loss(NLL): {}".format(epoch, test_loss))
				buffer = "Pre-train Discriminator Epoch:\t{}\tNLL:\t{}\n".format(str(epoch), str(test_loss))
				log.write(buffer)
				log.flush()

		dis_norm = self.normalize(dis, temp_min, temp_max)
		self.matplotformat(ax2, dis_norm, 'Pre-train Discriminator', pm.D_PRE_TRAIN_EPOCH)

# End of pre-train Discriminator --------------------

# Adversarial training between Generator and Discriminator --------------

# Generator update(freezing Discriminator while updating Generator and Monte-Carlo Model one time per epoch) ----------

		# reinforcement = Reinforcement(generator, pm.UPDATE_RATE)

		# Adversarial Train
		print("MSG : Start Adversarial Training...")
		log.write("Adversarial Training...\n")
		log.flush()
		for total_batch in range(pm.TOTAL_BATCHES):
			# Train the generator for one step
			for i in range(pm.G_STEP):
				# samples = generator.generate(sess)
				# rewards = reinforcement.get_reward(sess, samples, pm.MONTE_CARLO_TURNS, discriminator)
				# start_time = time()
				batch = gen_data_loader.next_batch()
				samples = attention_mechanism.generate(sess, batch, [pm.SEQ_LENGTH for _ in range(gen_data_loader.batch_size)])
				rewards = attention_mechanism.get_reward(sess, samples, batch, [pm.SEQ_LENGTH for _ in range(gen_data_loader.batch_size)],
														discriminator, pm.ADVERSARIAL_DROPOUT)

				# rewards = attention_mechanism.get_reward_multiterms(sess, samples, batch, [pm.SEQ_LENGTH for _ in range(gen_data_loader.batch_size)],
				# 													discriminator, pm.ADVERSARIAL_DROPOUT)
				# _ = sess.run(generator.g_updates, feed_dict={generator.x: samples, generator.rewards: rewards})
				_ = attention_mechanism.update_params(sess, samples, [pm.SEQ_LENGTH for _ in range(gen_data_loader.batch_size)], samples, rewards)
				# end_time = time()
				# print(end_time - start_time)

				# Teacher Forcing
				if pm.TEACHER_FORCING:
					real_rewards = np.array([[1. for _ in range(pm.SEQ_LENGTH)] for _ in range(gen_data_loader.batch_size)])
					attention_mechanism.update_params(sess, batch, [pm.SEQ_LENGTH for _ in range(gen_data_loader.batch_size)], batch, real_rewards)

			# Testing the result for each batch
			if total_batch % 5 == 0 or total_batch == pm.TOTAL_BATCHES - 1:
				# self.generate_samples(sess, generator, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.ADVERSARIAL_G_DATA)
				self.att_generate_samples(sess, attention_mechanism, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.ADVERSARIAL_G_DATA, gen_data_loader)
				likelihood_data_loader.mini_batch(pm.ADVERSARIAL_G_DATA)
				test_loss = self.target_loss(sess, corpus_lstm, likelihood_data_loader)
				learning_rate = sess.run(attention_mechanism.learning_rate_decay, feed_dict={attention_mechanism.global_step: total_batch})
				# test_loss = self.target_loss(sess, generator, likelihood_data_loader)
				gen.append(test_loss)
				buffer = "Adversarial Epoch:\t{}\tG_NLL:\t{}\tLearning_rate:\t{}\n".format(str(total_batch), str(test_loss), str(learning_rate))
				print("Adversarial Epoch: [{}/{}], GEN_Test_loss(NLL): {}, Learning_rate: {}".format(total_batch, pm.TOTAL_BATCHES, test_loss, learning_rate))
				log.write(buffer)
				log.flush()

			# Update reinforcement parameters
			
			# reinforcement.update_params()

# End of updating Generator ---------------

# Discriminator update(freezing Generator while updating Discriminator five times per epoch) -------------

			# Train the discriminator for five step
			for i in range(pm.D_STEP):
				# self.generate_samples(sess, generator, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.ADVERSARIAL_NEG_DATA)
				# self.dis_generate_samples(sess, attention_mechanism, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.ADVERSARIAL_NEG_DATA, gen_data_loader)
				self.att_generate_samples(sess, attention_mechanism, pm.BATCH_SIZE, pm.GENERATED_NUM, pm.ADVERSARIAL_NEG_DATA, gen_data_loader)
				dis_data_loader.mini_batch(pm.REAL_DATA_PATH, pm.ADVERSARIAL_NEG_DATA)
				for _ in range(pm.K):
					test_loss = self.dis_pre_train_loss(sess, discriminator, dis_data_loader)
					if test_loss < temp_min:
						temp_min = test_loss
					if test_loss > temp_max:
						temp_max = test_loss

			if total_batch % 5 == 0 or total_batch == pm.TOTAL_BATCHES - 1:
				dis.append(test_loss)
				print("Adversarial Epoch: [{}/{}], DIS_Test_loss(NLL): {}".format(total_batch, pm.TOTAL_BATCHES, test_loss))
				buffer = "Adversarial Epoch:\t{}\tD_NLL:\t{}\n".format(str(total_batch), str(test_loss))
				log.write(buffer)
				log.flush()

		# end_time = time()
		# time_token = end_time - start_time
		# print(time_token)
		# buffer = "Attention time token:\t{}\n".format(str(time_token))
		# log.write(buffer)
		# log.flush()

		ad_dis_norm = self.normalize(dis, temp_min, temp_max)
		self.matplotformat(ax3, gen, 'Adversarial Generator', pm.ATTENTION_PRE_TRAIN_EPOCH + pm.TOTAL_BATCHES)
		self.matplotformat(ax4, ad_dis_norm, 'Adversarial Discriminator', pm.D_PRE_TRAIN_EPOCH + pm.TOTAL_BATCHES)
		plt.tight_layout()
		plt.show()

		log.close()

		# word2idx, idx2word = chinese_quatrains_data_loader.load_vocabulary(pm.CHINESE_VOCAB_FIVE)
		# self.translate(idx2word, pm.CHINESE_5_TRANSLATE, pm.ADVERSARIAL_G_DATA)
		print("MSG : Done!")

# End of updating Discriminator -----------------

	def generate_samples(self, sess, trainable_model, batch_size, generated_num, output_path, dis_output_path):
		generated_samples = []
		# dis_out = codecs.open(dis_output_path, 'w', encoding='utf-8')
		total_num = generated_num // batch_size
		for _ in range(total_num):
			generated_samples.extend(trainable_model.generate(sess))

		with codecs.open(output_path, 'w', encoding='utf-8') as fout:
			for data in generated_samples:
				buffer = " ".join(str(x) for x in data)  # write in token format
				fout.write(buffer + '\n')

				# write mask token
				# for given_num in range(1, pm.SEQ_LENGTH + 1):
				# 	mask = [1 for _ in range(given_num)] + [0 for _ in range(pm.SEQ_LENGTH - given_num)]
				# 	temp_buffer = " ".join(str(x) for x in data * mask)
				# 	dis_out.write(temp_buffer + '\n')

	def att_generate_samples(self, sess, trainable_model, batch_size, generated_num, output_path, data_loader):
		att_generated_samples = []
		total_num = generated_num // batch_size
		for _ in range(total_num):
			batch = data_loader.next_batch()
			att_generated_samples.extend(trainable_model.generate(sess, batch, [pm.SEQ_LENGTH for _ in range(data_loader.batch_size)]))
		with codecs.open(output_path, 'w', encoding='utf-8') as fout:
			for data in att_generated_samples:
				buffer = " ".join(str(x) for x in data)
				fout.write(buffer + '\n')

	def dis_generate_samples(self, sess, trainable_model, batch_size, generated_num, output_path, data_loader):
		dis_generated_samples = []
		total_num = generated_num // batch_size
		for _ in range(total_num):
			batch = data_loader.next_batch()
			dis_generated_samples.extend(trainable_model.generate(sess, batch, [pm.SEQ_LENGTH for _ in range(data_loader.batch_size)]))
		with codecs.open(output_path, 'w', encoding='utf-8') as fout:
			for data in dis_generated_samples:
				# write mask token
				for given_num in range(1, pm.SEQ_LENGTH + 1):
					mask = [1 for _ in range(given_num)] + [0 for _ in range(pm.SEQ_LENGTH - given_num)]
					temp_buffer = " ".join(str(x) for x in data * mask)
					fout.write(temp_buffer + '\n')

	def target_loss(self, sess, target_model, data_loader):
		nll = []
		data_loader.reset_pointer()

		for i in range(data_loader.num_batch):
			batch = data_loader.next_batch()
			loss = sess.run(target_model.loss, feed_dict={target_model.x: batch})
			nll.append(loss)

		return np.mean(nll)

	def gen_pre_train_loss(self, sess, trainable_model, data_loader):
		supervised_g_losses = []
		data_loader.reset_pointer()

		for i in range(data_loader.num_batch):
			batch = data_loader.next_batch()
			_, g_loss = trainable_model.pretrain_forward(sess, batch)
			supervised_g_losses.append(g_loss)

		return np.mean(supervised_g_losses)

	def att_pre_train_loss(self, sess, trainable_model, data_loader):
		supervised_att_losses = []
		data_loader.reset_pointer()

		for i in range(data_loader.num_batch):
			batch = data_loader.next_batch()
			_, att_loss = trainable_model.pretrain_forward(sess, batch, [pm.SEQ_LENGTH for _ in range(data_loader.batch_size)], batch)
			supervised_att_losses.append(att_loss)

		return np.mean(supervised_att_losses)

	def dis_pre_train_loss(self, sess, trainable_model, data_loader):
		supervised_d_losses = []
		data_loader.reset_pointer()

		for i in range(data_loader.num_batch):
			x_batch, y_batch = data_loader.next_batch()
			_, d_loss = trainable_model.pretrain_forward(sess, x_batch, y_batch, pm.D_DROP_KEEP_PROB)
			supervised_d_losses.append(d_loss)

		return np.mean(supervised_d_losses)

	def normalize(self, temp_list, temp_min, temp_max):
		output = [((i - temp_min) / (temp_max - temp_min)) for i in temp_list]
		return output

	def matplotformat(self, ax, plot_y, plot_name, x_max):
		plt.sca(ax)
		plot_x = [i * 5 for i in range(len(plot_y))]
		plt.xticks(np.linspace(0, x_max, (x_max // 100) + 1, dtype=np.int32))
		plt.xlabel('Epochs', fontsize=16)
		plt.ylabel('NLL by oracle', fontsize=16)
		plt.title(plot_name)
		plt.plot(plot_x, plot_y)

	def translate(self, vocabulary, writefile, loadfile):
		fout = codecs.open(writefile, 'w', encoding='utf-8')
		with codecs.open(loadfile, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip('\n')
				sentence = " ".join(vocabulary.get(int(token), 1) for token in line.split() if token != '3')
				fout.write(sentence + '\n')
				fout.flush()
		fout.close()

if __name__ == '__main__':
	model = SeqGAN()
