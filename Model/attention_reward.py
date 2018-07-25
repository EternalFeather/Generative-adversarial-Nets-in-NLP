# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib.rnn import LSTMCell
from Config.hyperparameters import Parameters as pm


class Attention_reward(object):
	def __init__(self, vocab_size, batch_size, embed_size, hidden_size, sequence_length, start_token, learning_rate, decay_steps, staricase):
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.sequence_length = sequence_length
		self.dec_sequence_length = sequence_length
		self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
		self.learning_rate = learning_rate
		self.decay_steps = decay_steps
		self.staricase = staricase
		self.grad_clip = 5.0
		self.params = []

		with tf.variable_scope('attention_reward_enc'):
			self.enc_embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.embed_size]))
			self.params.append(self.enc_embeddings)
			# self.enc_lstm_forward = self.enc_recurrent_lstm_forward(self.params)

		with tf.variable_scope('attention_reward_dec'):
			self.dec_embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.embed_size]))
			self.params.append(self.dec_embeddings)
			self.dec_lstm_forward = self.dec_recurrent_lstm_forward(self.params)
			self.dec_linear_forward = self.dec_recurrent_linear_forward(self.params)

		with tf.variable_scope('attention_reward_att'):
			self.attention_forward = self.recurrent_alignment_forward(self.params)

# Initialize parameters ------------------

		self.enc_x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
		self.enc_sequence_length = tf.placeholder(tf.int32, shape=[self.batch_size])
		self.dec_x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length])
		self.rewards = tf.placeholder(tf.float32, shape=[self.batch_size, self.dec_sequence_length])

		with tf.device("/cpu:0"):
			self.processed_enc_x = tf.transpose(tf.nn.embedding_lookup(self.enc_embeddings, self.enc_x), perm=[1, 0, 2])
			self.processed_dec_x = tf.transpose(tf.nn.embedding_lookup(self.dec_embeddings, self.dec_x), perm=[1, 0, 2])

		self.enc_h0 = tf.zeros([self.batch_size, self.hidden_size])
		self.enc_h0 = tf.stack([self.enc_h0, self.enc_h0])
		self.enc_single_h0 = tf.zeros([self.batch_size, self.hidden_size])
		self.dec_s0 = tf.zeros([self.batch_size, self.hidden_size])
		self.dec_s0 = tf.stack([self.dec_s0, self.dec_s0])
		self.dec_single_s0 = tf.zeros([self.batch_size, self.hidden_size])

		self.global_step = tf.Variable(tf.constant(0))
		self.learning_rate_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, 0.96, staircase=self.staricase)

		# enc_hiddens = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
		#                                           dynamic_size=False, infer_shape=True)
		# enc_embed_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
		#                                           dynamic_size=False, infer_shape=True)
		# enc_embed_x = enc_embed_x.unstack(self.processed_enc_x)

		dec_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)
		dec_token_sequence = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)
		dec_token_sequence = dec_token_sequence.unstack(self.processed_dec_x)
		alphas_sequence = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)


# End initialize ------------------------

# Pre-train attention step ------------------------

# 		def _enc_pretrain_recurrence(i, x_t, h_tm, state, hiddens):
# 			state, h_t = self.enc_lstm_forward(x_t, h_tm)
# 			hiddens.write(i, state)
# 			x_ = enc_embed_x.read(i)
# 			return i + 1, x_, h_t, state, hiddens
#
# 		_, _, _, _, self.enc_hiddens = control_flow_ops.while_loop(
# 			cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
# 			body=_enc_pretrain_recurrence,
# 			loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.enc_embeddings, self.start_token),
# 						self.enc_h0, self.enc_single_h0, enc_hiddens)
# 		)   # shape = [batch_size, hidden_size]
#
# 		self.enc_hiddens = self.enc_hiddens.stack()
# 		self.enc_hiddens = tf.transpose(self.enc_hiddens, perm=[1, 0, 2])
		if pm.BiLSTM:
			self.enc_hiddens, _ = bidirectional_dynamic_rnn(LSTMCell(self.hidden_size), LSTMCell(self.hidden_size),
															inputs=tf.nn.embedding_lookup(self.enc_embeddings, self.enc_x),
															sequence_length=self.enc_sequence_length, dtype=tf.float32)
			self.enc_hiddens = tf.concat(self.enc_hiddens, 2)   # shape = [batch_size, sequence_length, hidden_size]
		else:
			self.enc_hiddens, _ = dynamic_rnn(LSTMCell(self.hidden_size),
											inputs=tf.nn.embedding_lookup(self.enc_embeddings, self.enc_x),
											sequence_length=self.enc_sequence_length, dtype=tf.float32)

		def _dec_attention_pretrain_recurrence(i, x_t, s_t, s_tm, enc_hiddens, dec_predictions, alpha_seq):
			hiddens = tf.tile(tf.expand_dims(s_t, 1), [1, self.sequence_length, 1])     # shape = [batch_size, sequence_length, hidden_size]
			context, alphas = self.attention_forward(enc_hiddens, hiddens)
			alpha_seq = alpha_seq.write(i, alphas)      # shape = [dec_sequence_length, batch_size, enc_sequence_length]
			s_t, s_tm = self.dec_lstm_forward(x_t, s_tm, context)
			o_t = self.dec_linear_forward(s_tm)
			dec_predictions = dec_predictions.write(i, o_t)     # softmax_distribution, shape = [batch_size, vocab_size]
			x_ = dec_token_sequence.read(i)
			return i + 1, x_, s_t, s_tm, enc_hiddens, dec_predictions, alpha_seq

		_, _, _, _, _, self.dec_predictions, self.alphas_sequence = control_flow_ops.while_loop(
			cond=lambda i, _1, _2, _3, _4, _5, _6: i < self.sequence_length,
			body=_dec_attention_pretrain_recurrence,
			loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.dec_embeddings, self.start_token),
						self.dec_single_s0, self.dec_s0, self.enc_hiddens, dec_predictions, alphas_sequence)
		)       # shape = [sequence_length, batch_size, vocab_size]

		self.dec_predictions = self.dec_predictions.stack()
		self.dec_predictions = tf.transpose(self.dec_predictions, perm=[1, 0, 2])   # shape = [batch_size, sequence_length, vocab_size]

		self.alphas_sequence = self.alphas_sequence.stack()
		self.alphas_sequence = tf.transpose(self.alphas_sequence, perm=[1, 0, 2])   # shape = [batch_size, dec_sequence_length, enc_sequence_length]

# End pre-train attention step --------------------

# Generate step -----------------------------------

		output_prob_sequence = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length, dynamic_size=False, infer_shape=True)
		token_sequence = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length, dynamic_size=False, infer_shape=True)

		def _generate_recurrence(i, x_t, s_t, s_tm, enc_hiddens, gen_o, gen_x):
			hiddens = tf.tile(tf.expand_dims(s_t, 1), [1, self.sequence_length, 1])
			context, _ = self.attention_forward(enc_hiddens, hiddens)
			s_t, s_tm = self.dec_lstm_forward(x_t, s_tm, context)
			o_t = self.dec_linear_forward(s_tm)
			log_prob = tf.log(o_t)
			next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
			x_ = tf.nn.embedding_lookup(self.dec_embeddings, next_token)
			gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0), o_t), 1))
			gen_x = gen_x.write(i, next_token)
			return i + 1, x_, s_t, s_tm, enc_hiddens, gen_o, gen_x

		_, _, _, _, _, self.output_prob_sequence, self.token_sequence = control_flow_ops.while_loop(
			cond=lambda i, _1, _2, _3, _4, _5, _6: i < self.sequence_length,
			body=_generate_recurrence,
			loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.dec_embeddings, self.start_token),
						self.dec_single_s0, self.dec_s0, self.enc_hiddens, output_prob_sequence, token_sequence)
		)

		self.token_sequence = self.token_sequence.stack()
		self.token_sequence = tf.transpose(self.token_sequence, perm=[1, 0])    # shape = [batch_size, sequence_length]
		
# End Generate step -------------------------------

# Pre-train Compile configuration -----------------

		# pre-train loss(cross-entropy between enc_x and dec_x(predict))
		self.loss = -tf.reduce_sum(
			tf.one_hot(tf.cast(tf.reshape(self.dec_x, [-1]), tf.int32), self.vocab_size, 1.0, 0.0) * tf.log(
				tf.clip_by_value(tf.reshape(self.dec_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
			)
		) / (self.sequence_length * self.batch_size)

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

# End Setting Pre-train Compile -------------------

# Training pre-train model ------------------------

		self.pretrain_grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.params), self.grad_clip)
		self.pretrain_updates = self.optimizer.apply_gradients(zip(self.pretrain_grad, self.params))

# End training pre-train --------------------------

# Adversarial Learning train ----------------

		# output is a number represents the whole reward of sequence_tokens
		self.ad_loss = -tf.reduce_sum(
			tf.reduce_sum(
				tf.one_hot(tf.cast(tf.reshape(self.dec_x, [-1]), tf.int32), self.vocab_size, 1.0, 0.0) * tf.log(
					tf.clip_by_value(tf.reshape(self.dec_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
				), 1) * tf.reshape(self.rewards, [-1])
		)  # Adversarial Learning with rewards, [seq_length * batch_size] * (sum of prob)
		# ==> sum([seq_length * batch_size] * (sum of prob) * (rewards))

		if pm.DYNAMIC_LR:
			self.ad_optimizer = tf.train.AdamOptimizer(self.learning_rate_decay)
		else:
			self.ad_optimizer = tf.train.AdamOptimizer(self.learning_rate)

		self.ad_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.ad_loss, self.params), self.grad_clip)
		self.ad_updates = self.ad_optimizer.apply_gradients(zip(self.ad_gradients, self.params), global_step=self.global_step)

# End Adversarial Learning ------------------

	def init_matrix(self, shape):
		return tf.random_normal(shape, stddev=0.1)

	# def enc_recurrent_lstm_forward(self, params):
	# 	self.Wi = tf.Variable(self.init_matrix([self.embed_size, self.hidden_size]))
	# 	self.Ui = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
	# 	self.bi = tf.Variable(self.init_matrix([self.hidden_size]))
	#
	# 	self.Wf = tf.Variable(self.init_matrix([self.embed_size, self.hidden_size]))
	# 	self.Uf = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
	# 	self.bf = tf.Variable(self.init_matrix([self.hidden_size]))
	#
	# 	self.Wo = tf.Variable(self.init_matrix([self.embed_size, self.hidden_size]))
	# 	self.Uo = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
	# 	self.bo = tf.Variable(self.init_matrix([self.hidden_size]))
	#
	# 	self.Wc = tf.Variable(self.init_matrix([self.embed_size, self.hidden_size]))
	# 	self.Uc = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
	# 	self.bc = tf.Variable(self.init_matrix([self.hidden_size]))
	#
	# 	params.extend([
	# 		self.Wi, self.Ui, self.bi,
	# 		self.Wf, self.Uf, self.bf,
	# 		self.Wo, self.Uo, self.bo,
	# 		self.Wc, self.Uc, self.bc]
	# 	)
	#
	# 	def forward(x, hidden_memory):
	# 		hidden_state, cell = tf.unstack(hidden_memory)
	#
	# 		i = tf.sigmoid(
	# 			tf.matmul(x, self.Wi) +
	# 			tf.matmul(hidden_state, self.Ui) + self.bi
	# 		)
	#
	# 		f = tf.sigmoid(
	# 			tf.matmul(x, self.Wf) +
	# 			tf.matmul(hidden_state, self.Uf) + self.bf
	# 		)
	#
	# 		o = tf.sigmoid(
	# 			tf.matmul(x, self.Wo) +
	# 			tf.matmul(hidden_state, self.Uo) + self.bo
	# 		)
	#
	# 		c_ = tf.nn.tanh(
	# 			tf.matmul(x, self.Wc) +
	# 			tf.matmul(hidden_state, self.Uc) + self.bc
	# 		)
	#
	# 		c = f * cell + i * c_
	#
	# 		current_hidden_state = o * tf.nn.tanh(c)
	#
	# 		return current_hidden_state, tf.stack([current_hidden_state, c])
	#
	# 	return forward

	def dec_recurrent_lstm_forward(self, params):
		self.Wi = tf.Variable(self.init_matrix([self.embed_size, self.hidden_size]))
		self.Ui = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		if pm.BiLSTM:
			self.Ci = tf.Variable(self.init_matrix([2 * self.hidden_size, self.hidden_size]))
		else:
			self.Ci = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.bi = tf.Variable(self.init_matrix([self.hidden_size]))

		self.Wf = tf.Variable(self.init_matrix([self.embed_size, self.hidden_size]))
		self.Uf = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		if pm.BiLSTM:
			self.Cf = tf.Variable(self.init_matrix([2 * self.hidden_size, self.hidden_size]))
		else:
			self.Cf = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.bf = tf.Variable(self.init_matrix([self.hidden_size]))

		self.Wo = tf.Variable(self.init_matrix([self.embed_size, self.hidden_size]))
		self.Uo = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		if pm.BiLSTM:
			self.Co = tf.Variable(self.init_matrix([2 * self.hidden_size, self.hidden_size]))
		else:
			self.Co = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.bo = tf.Variable(self.init_matrix([self.hidden_size]))

		self.Wc = tf.Variable(self.init_matrix([self.embed_size, self.hidden_size]))
		self.Uc = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		if pm.BiLSTM:
			self.Cc = tf.Variable(self.init_matrix([2 * self.hidden_size, self.hidden_size]))
		else:
			self.Cc = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.bc = tf.Variable(self.init_matrix([self.hidden_size]))

		params.extend([
			self.Wi, self.Ui, self.Ci, self.bi,
			self.Wf, self.Uf, self.Cf, self.bf,
			self.Wo, self.Uo, self.Co, self.bo,
			self.Wc, self.Uc, self.Cc, self.bc]
		)

		def forward(x, hidden_memory, context):
			hidden_state, cell = tf.unstack(hidden_memory)

			i = tf.sigmoid(
				tf.matmul(x, self.Wi) +
				tf.matmul(hidden_state, self.Ui) + tf.matmul(context, self.Ci) + self.bi
			)

			f = tf.sigmoid(
				tf.matmul(x, self.Wf) +
				tf.matmul(hidden_state, self.Uf) + tf.matmul(context, self.Cf) + self.bf
			)

			o = tf.sigmoid(
				tf.matmul(x, self.Wo) +
				tf.matmul(hidden_state, self.Uo) + tf.matmul(context, self.Co) + self.bo
			)

			c_ = tf.nn.tanh(
				tf.matmul(x, self.Wc) +
				tf.matmul(hidden_state, self.Uc) + tf.matmul(context, self.Cc) + self.bc
			)

			c = f * cell + i * c_

			current_hidden_state = o * tf.nn.tanh(c)

			return current_hidden_state, tf.stack([current_hidden_state, c])

		return forward

	def dec_recurrent_linear_forward(self, params):
		self.V = tf.Variable(self.init_matrix([self.hidden_size, self.vocab_size]))
		self.c = tf.Variable(self.init_matrix([self.vocab_size]))

		params.extend([
			self.V, self.c]
		)

		def forward(hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)
			logits = tf.matmul(hidden_state, self.V) + self.c
			output = tf.nn.softmax(logits)
			return output

		return forward

	def recurrent_alignment_forward(self, params):
		if pm.BiLSTM:
			self.W_alpha = tf.Variable(self.init_matrix([2 * self.hidden_size, self.hidden_size]))
		else:
			self.W_alpha = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.U_alpha = tf.Variable(self.init_matrix([self.hidden_size, self.hidden_size]))
		self.b_alpha = tf.Variable(self.init_matrix([self.hidden_size]))
		self.V_alpha = tf.Variable(self.init_matrix([self.hidden_size]))

		params.extend([
			self.W_alpha, self.U_alpha, self.b_alpha, self.V_alpha]
		)

		def forward(h, s):
			a_ = tf.nn.tanh(tf.tensordot(h, self.W_alpha, axes=[[2], [0]]) +
							tf.tensordot(s, self.U_alpha, axes=[[2], [0]]) +
							self.b_alpha)   # shape = [batch_size, sequence_length, hidden_size]

			a = tf.tensordot(a_, self.V_alpha, axes=[[2], [0]])     # shape = [batch_size, sequence_length]
			alphas = tf.nn.softmax(a)   # shape = [batch_size, sequence_length]

			context_vector = tf.reduce_sum(h * tf.expand_dims(alphas, -1), 1)    # shape = [batch_size, hidden_size]

			return context_vector, alphas

		return forward

	def pretrain_forward(self, sess, x1, seq, x2):
		outputs = sess.run([self.pretrain_updates, self.loss], feed_dict={self.enc_x: x1, self.enc_sequence_length: seq, self.dec_x: x2})
		return outputs

	def generate(self, sess, x, seq):
		output = sess.run(self.token_sequence, feed_dict={self.enc_x: x, self.enc_sequence_length: seq})
		return output

	def get_reward(self, sess, x, y, seq_length, discriminator, drop_rate):
		rewards = []

		ypred_for_acu = sess.run(discriminator.y_pred_for_acu, feed_dict={discriminator.x: x, discriminator.dropout_keep_prob: drop_rate})
		ypred = np.array([item[1] for item in ypred_for_acu])   # shape = [batch_size]
		pred_rewards = np.array([[(reward / self.sequence_length) for _ in range(self.sequence_length)] for reward in ypred])   # shape = [batch_size, enc_sequence_length]

		# x(64, 20), y(64, 20), alpha(64, 20, 20)
		# shape = [batch_size, dec_sequence_length, enc_sequence_length]
		alphas = sess.run(self.alphas_sequence, feed_dict={self.enc_x: x, self.enc_sequence_length: seq_length, self.dec_x: y})

		batch_size = alphas.shape[0]
		for batch in range(batch_size):
			step_rewards = np.dot(self.sequence_length * alphas[batch], pred_rewards[batch])
			rewards.append(step_rewards)

		return np.array(rewards)    # shape = [bach_size, dec_sequence_length]

	def get_reward_multiterms(self, sess, x, y, seq_length, discriminator, drop_rate):
		pred_sets, step_rewards, rewards = [], [], []
		alphas = sess.run(self.alphas_sequence, feed_dict={self.enc_x: x, self.enc_sequence_length: seq_length, self.dec_x: y})
		# shape = [batch_size, dec_sequence_length, enc_sequence_length]

		for given_num in range(1, pm.SEQ_LENGTH + 1):
			mask = [[1 for _ in range(given_num)] + [0 for _ in range(pm.SEQ_LENGTH - given_num)] for _ in range(pm.BATCH_SIZE)]
			sampled_x = x * mask
			ypred_for_acu = sess.run(discriminator.y_pred_for_acu, feed_dict={discriminator.x: sampled_x, discriminator.dropout_keep_prob: drop_rate})
			ypred = np.array([item[1] for item in ypred_for_acu])   # shape = [batch_size]
			pred_sets.append(ypred)   # shape = [sequence_length, batch_size]

		pred_sets = np.transpose(np.array(pred_sets))   # shape = [batch_size, sequence_length]

		for batch in range(pm.BATCH_SIZE):
			for i in range(pm.SEQ_LENGTH):
				if i == 0:
					step_rewards = np.multiply(pred_sets[batch][i], alphas[batch, i, :])
				else:
					step_rewards += np.multiply(pred_sets[batch][i], alphas[batch, i, :])

			rewards.append(step_rewards / pm.SEQ_LENGTH)    # normalize
		rewards = np.array(rewards)
		return rewards

	def update_params(self, sess, x1, seq, x2, rewards):
		ls = sess.run([self.ad_loss, self.ad_updates], feed_dict={self.enc_x: x1, self.enc_sequence_length: seq, self.dec_x: x2, self.rewards: rewards})
		return ls
