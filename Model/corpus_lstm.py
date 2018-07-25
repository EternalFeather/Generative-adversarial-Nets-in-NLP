# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class Corpus_lstm(object):
	def __init__(self, vocab_size, batch_size, emb_size, hidden_size, seq_length, start_token, params):
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.seq_length = seq_length
		self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
		self.params = params
		self.target_params = []
		tf.set_random_seed(66)

		with tf.variable_scope("target_lstm"):
			self.target_embeddings = tf.Variable(self.params[0])
			self.target_params.append(self.target_embeddings)
			self.target_lstm_forward = self.recurrent_lstm_forward(self.target_params)
			self.target_linear_forward = self.recurrent_linear_forward(self.target_params)

# Initialize parameters ------------------

		# placeholder
		self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_length])

		# processed for input(real_data --> for pre-train)
		with tf.device("/cpu:0"):
			self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.target_embeddings, self.x), perm=[1, 0, 2]) # shape=[seq_length, batch_size, emb_size]

		# Initialize hidden_state + cell
		self.h0 = tf.zeros([self.batch_size, self.hidden_size])
		self.h0 = tf.stack([self.h0, self.h0])

		# Initialize target_lstm sequence
		target_prob = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_length, dynamic_size=False, infer_shape=True)
		target_token_sequence = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.seq_length, dynamic_size=False, infer_shape=True)

# End initialize ------------------

# Forward step -------------------

		def _target_recurrence(i, x_t, h_tm, gen_o, gen_x):
			h_t = self.target_lstm_forward(x_t, h_tm)
			o_t = self.target_linear_forward(h_t)
			log_prob = tf.log(o_t)
			next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
			x_ = tf.nn.embedding_lookup(self.target_embeddings, next_token)
			gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0), o_t), 1))
			gen_x = gen_x.write(i, next_token)
			return i + 1, x_, h_t, gen_o, gen_x

		_, _, _, self.target_prob, self.target_token_sequence = control_flow_ops.while_loop(
			cond=lambda i, _1, _2, _3, _4: i < self.seq_length,
			body=_target_recurrence,
			loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.target_embeddings, self.start_token),
						self.h0, target_prob, target_token_sequence)
		)

		self.target_token_sequence = self.target_token_sequence.stack()
		self.target_token_sequence = tf.transpose(self.target_token_sequence, perm=[1, 0])  # Output a token using softmax distribution random choice, shape = [batch_size, seq_length]

# End Forward step ----------------------

# Pre-train step -------------------------

		# Pre-train for target-lstm
		target_predictions = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_length, dynamic_size=False, infer_shape=True)
		real_x = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.seq_length, dynamic_size=False, infer_shape=True)
		real_x = real_x.unstack(self.processed_x)   # Gain real data's token embedding

		def _pretrain_recurrence(i, x_t, h_tm, target_predictions):
			h_t = self.target_lstm_forward(x_t, h_tm)
			o_t = self.target_linear_forward(h_t)
			target_predictions = target_predictions.write(i, o_t)
			x_ = real_x.read(i)
			return i + 1, x_, h_t, target_predictions

		_, _, _, self.target_predictions = control_flow_ops.while_loop(
			cond=lambda i, _1, _2, _3: i < self.seq_length,
			body=_pretrain_recurrence,
			loop_vars=(tf.constant(0, dtype=tf.int32), tf.nn.embedding_lookup(self.target_embeddings, self.start_token),
						self.h0, target_predictions)
		)

		self.target_predictions = self.target_predictions.stack()
		self.target_predictions = tf.transpose(self.target_predictions, perm=[1, 0, 2])    # Output a softmax distribution, shape = [batch_size, seq_length, vocab_size]

# End pre-train step -------------------

# Pre-train Loss configuration ----------------

		# Pre-train loss
		self.loss = -tf.reduce_sum(
			tf.one_hot(tf.cast(tf.reshape(self.x, [-1]), tf.int32), self.vocab_size, 1.0, 0.0) * tf.log(
				tf.reshape(self.target_predictions, [-1, self.vocab_size])
			)
		) / (self.seq_length * self.batch_size)

# End Setting Pre-train Loss ------------------

# Adversarial Loss configuration ------------------

		# Adversarial learning loss
		self.target_loss = tf.reduce_sum(
			tf.reshape(
				-tf.reduce_sum(
					tf.one_hot(tf.cast(tf.reshape(self.x, [-1]), tf.int32), self.vocab_size, 1.0, 0.0) * tf.log(
						tf.reshape(self.target_predictions, [-1, self.vocab_size])
					), 1    # shape = [seq_length * batch_size, 1]
				), [-1, self.seq_length]    # shape = [batch_size, seq_length]
			), 1    # loss for batch_sequence, shape = [batch_size]
		)

# End Adversarial Loss ----------------------------

	def generate(self, sess):
		outputs = sess.run(self.target_token_sequence)
		return outputs

	def recurrent_lstm_forward(self, params):
		self.Wi = tf.Variable(self.params[1])
		self.Ui = tf.Variable(self.params[2])
		self.bi = tf.Variable(self.params[3])

		self.Wf = tf.Variable(self.params[4])
		self.Uf = tf.Variable(self.params[5])
		self.bf = tf.Variable(self.params[6])

		self.Wo = tf.Variable(self.params[7])
		self.Uo = tf.Variable(self.params[8])
		self.bo = tf.Variable(self.params[9])

		self.Wc = tf.Variable(self.params[10])
		self.Uc = tf.Variable(self.params[11])
		self.bc = tf.Variable(self.params[12])
		params.extend([
			self.Wi, self.Ui, self.bi,
			self.Wf, self.Uf, self.bf,
			self.Wo, self.Uo, self.bo,
			self.Wc, self.Uc, self.bc]
		)

		def forward(x, hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)

			i = tf.sigmoid(
				tf.matmul(x, self.Wi) + tf.matmul(hidden_state, self.Ui) + self.bi
			)

			f = tf.sigmoid(
				tf.matmul(x, self.Wf) + tf.matmul(hidden_state, self.Uf) + self.bf
			)

			o = tf.sigmoid(
				tf.matmul(x, self.Wo) + tf.matmul(hidden_state, self.Uo) + self.bo
			)

			c_ = tf.nn.tanh(
				tf.matmul(x, self.Wc) + tf.matmul(hidden_state, self.Uc) + self.bc
			)

			c = f * cell + i * c_

			current_hidden_state = o * tf.nn.tanh(c)

			return tf.stack([current_hidden_state, c])

		return forward

	def recurrent_linear_forward(self, params):
		self.V = tf.Variable(self.params[13])
		self.c = tf.Variable(self.params[14])
		params.extend([self.V, self.c])

		def forward(hidden_memory):
			hidden_state, cell = tf.unstack(hidden_memory)
			logits = tf.matmul(hidden_state, self.V) + self.c
			output = tf.nn.softmax(logits)
			return output

		return forward
