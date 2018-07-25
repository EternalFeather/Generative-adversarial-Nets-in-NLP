# -*- coding: utf-8 -*-
import tensorflow as tf
from Transformer.corpora.data_loader import Data_helper
from Transformer.config.hyperparams import Hyperparams as pm
from Transformer.modules import Model


class Transformer(object):
	def __init__(self, trainable=True):
		self.models = Model()
		self.data_helper = Data_helper()

		if trainable:
			self.x, self.y, self.num_batch = self.data_helper.mini_batch()
		else:
			# Re-initialize
			self.x = tf.placeholder(tf.int32, shape=(None, pm.maxlen))
			self.y = tf.placeholder(tf.int32, shape=(None, pm.maxlen))

		# Add 2(<SOS>) in the beginning
		start_token = tf.ones_like(self.y[:, :1]) * 2
		self.decoder_inputs = tf.concat((start_token, self.y[:, :-1]), -1)

		# Load vocabulary
		self.de2idx, self.idx2de = self.data_helper.load_vocab(pm.DECODER_VOCAB)
		self.en2idx, self.idx2en = self.data_helper.load_vocab(pm.ENCODER_VOCAB)

# Module -----------------

		# Encoder
		with tf.variable_scope("encoder"):
			# Input Embedding
			self.encoder = self.models.embedding(self.x,
											vocab_size=len(self.de2idx),
											num_units=pm.hidden_units,
											scale=True,
											scope="Input_Embedding")

			# Positional Encoding
			if pm.sinusoid:
				self.encoder += self.models.positional_encoding(self.x,
																num_units=pm.hidden_units,
																zero_pad=False,
																scale=False,
																scope="enc_positional_encoding")
			else:
				self.encoder += self.models.embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
													vocab_size=pm.maxlen,
													num_units=pm.hidden_units,
													zero_pad=False,
													scale=False,
													scope="enc_positional_encoding")

			# Dropout
			self.encoder = tf.layers.dropout(self.encoder,
											rate=pm.dropout_rate,
											training=tf.convert_to_tensor(trainable))

			# Body_networks
			for num in range(pm.num_blocks):
				with tf.variable_scope("encoder_networds_{}".format(num)):
					# Multi-Head Attention
					self.encoder = self.models.multihead_attention(queries=self.encoder,
																keys=self.encoder,
																num_units=pm.hidden_units,
																num_heads=pm.num_heads,
																dropout_rate=pm.dropout_rate,
																trainable=trainable,
																mask=False)

					# Feed Forward
					self.encoder = self.models.feedforward(self.encoder, num_units=[4 * pm.hidden_units, pm.hidden_units])

		# Decoder
		with tf.variable_scope("decoder"):
			# Output Embedding
			self.decoder = self.models.embedding(self.decoder_inputs,
											vocab_size=len(self.en2idx),
											num_units=pm.hidden_units,
											scale=True,
											scope="Output_embedding")

			# Positional Encoding
			if pm.sinusoid:
				self.decoder += self.models.positional_encoding(self.decoder_inputs,
																num_units=pm.hidden_units,
																zero_pad=False,
																scale=False,
																scope="dec_Positional_Encoding")
			else:
				self.decoder += self.models.embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
													vocab_size=pm.maxlen,
													num_units=pm.hidden_units,
													zero_pad=False,
													scale=False,
													scope="dec_Positional_Encoding")

			# Dropout
			self.decoder = tf.layers.dropout(self.decoder,
											rate=pm.dropout_rate,
											training=tf.convert_to_tensor(trainable))

			# Body_networks
			for num in range(pm.num_blocks):
				with tf.variable_scope("decoder_networks_{}".format(num)):
					# Multi-Head Attention with Mask(self-attention)
					self.decoder = self.models.multihead_attention(queries=self.decoder,
																	keys=self.decoder,
																	num_units=pm.hidden_units,
																	num_heads=pm.num_heads,
																	dropout_rate=pm.dropout_rate,
																	trainable=trainable,
																	mask=True,
																	scope="dec_Multihead_Attention")

					# Multi-Head Attention(vanilla attention)
					self.decoder = self.models.multihead_attention(queries=self.decoder,
																	keys=self.encoder,
																	num_units=pm.hidden_units,
																	num_heads=pm.num_heads,
																	dropout_rate=pm.dropout_rate,
																	trainable=trainable,
																	mask=False,
																	scope="dec_Vanilla_Attention")

					# Feed Forward
					self.decoder = self.models.feedforward(self.decoder, num_units=[4 * pm.hidden_units, pm.hidden_units])

		# Linear & Softmax
		self.logits = tf.layers.dense(self.decoder, len(self.en2idx))
		self.predicts = tf.cast((tf.argmax(tf.nn.softmax(self.logits), dimension=-1)), tf.int32)
		self.is_target = tf.cast((tf.not_equal(self.y, 0)), tf.float32)
		self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predicts, self.y), tf.float32) * self.is_target) / (tf.reduce_sum(self.is_target))
		tf.summary.scalar('accuracy', self.accuracy)

# End Module ----------------

		# Compile
		if trainable:
			# Loss
			self.y_smoothed = self.models.label_smoothing(tf.one_hot(self.y, depth=len(self.en2idx)))
			self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
			self.mean_loss = tf.reduce_sum(self.loss * self.is_target) / (tf.reduce_sum(self.is_target))

			# Optimizer
			self.global_step = tf.Variable(0, name='global_step', trainable=False)  # when it is passed in the minimize() argument list ,the variable is increased by one
			self.optimizer = tf.train.AdamOptimizer(learning_rate=pm.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8).minimize(self.mean_loss, global_step=self.global_step)

			# Summary
			tf.summary.scalar('mean_loss', self.mean_loss)
			self.merged_summary = tf.summary.merge_all()
