# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self):
        self.model_num = 0

    def embedding(self, inputs, vocab_size, num_units, zero_pad=True, scale=True, scope="embedding", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            embeddings = tf.get_variable('embeddings',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units],
                                           initializer=tf.contrib.layers.xavier_initializer())

            if zero_pad:
                embeddings = tf.concat((tf.zeros(shape=[1, num_units]), embeddings[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(embeddings, inputs)

            if scale:
                outputs = outputs * tf.sqrt(tf.cast(num_units, tf.float32))

            self.model_num += 1

        return outputs

    def positional_encoding(self, inputs, num_units, zero_pad=True, scale=True, scope="positional_encoding", reuse=None):
        N, T = inputs.get_shape().as_list()
        with tf.variable_scope(scope, reuse=reuse):
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # shape = [batch_size, seq_length]

            # PE function for sin & cos embeddings
            position_encoder = np.array([[pos / np.power(10000, 2.0 * i / num_units) for i in range(num_units)] for pos in range(T)])
            position_encoder[:, 0::2] = np.sin(position_encoder[:, 0::2])
            position_encoder[:, 1::2] = np.cos(position_encoder[:, 1::2])

            # Convert to tensor
            embeddings = tf.convert_to_tensor(position_encoder)

            if zero_pad:
                embeddings = tf.concat((tf.zeros(shape=[1, num_units]), embeddings[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(embeddings, position_ind)

            if scale:
                outputs = outputs * tf.sqrt(tf.cast(num_units, tf.float32))

            self.model_num += 1

        return outputs

    def multihead_attention(self, queries, keys, num_units=None, num_heads=8, dropout_rate=0, trainable=True, mask=False, scope="multihead_attention", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            # Linear
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # shape = [N, T_q, C]
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # shape = [h * N, T_q, C / h]
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

# Scaled Dot-Product Attention ---------------

            # Attention(Q, K, V)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))    # shape = [h * N, T_q, T_k]

            # scale
            outputs = outputs / tf.sqrt(tf.cast(K_.get_shape().as_list()[-1], tf.float32))

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))   # shape = [N, T_k]
            key_masks = tf.tile(key_masks, [num_heads, 1])  # shape = [h * N, T_k]
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # shape = [h * N, T_q, T_k]

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)   # shape = [h * N, T_q, T_k]

            # Causality = Future blinding(opt.)
            if mask:
                diag_vals = tf.ones_like(outputs[0, :, :])  # shape = [T_q, T_k]
                # triangular linear operator
                # input : [[1,2,3], [4,5,6], [7,8,9]]
                # output : [[1,0,0], [4,5,0], [7,8,9]]
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # shape = [h * N, T_q, T_k]

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)   # shape = [h * N, T_q, T_k]

            # Softmax
            outputs = tf.nn.softmax(outputs)    # shape = [h * N, T_q, T_k]

            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # shape = [N, T_q]
            query_masks = tf.tile(query_masks, [num_heads, 1])    # shape = [N * h, T_q]
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])     # shape = [h * N, T_q, T_k]
            outputs *= query_masks   # shape = [N, T_q, C]

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(trainable))

            # Matmul
            outputs = tf.matmul(outputs, V_)    # shape = [h * N, T_q, C / h]

# End Scaled Dot-Product Attention -----------

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # shape = [N, T_q, C]

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)   # shape = [N, T_q, C]

            self.model_num += 1

        return outputs

    def feedforward(self, inputs, num_units, scope="feedforward", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.normalize(outputs)

            self.model_num += 1

        return outputs

    def label_smoothing(self, inputs, epsilon=0.1):
        K = inputs.get_shape().as_list()[-1]
        self.model_num += 1
        return ((1-epsilon) * inputs) + (epsilon / K)

    def normalize(self, inputs, epsilon=1e-8, scope="normalize", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / (tf.sqrt(tf.cast(variance + epsilon, tf.float32)))
            outputs = tf.multiply(normalized, gamma) + beta

            self.model_num += 1

        return outputs
