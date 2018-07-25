# -*- coding:utf-8 -*-
import tensorflow as tf
from Config.hyperparameters import Parameters as pm


class Discriminator(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embed_size, filter_sizes, num_filters, learning_rate, l2_reg_lambda):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda

# Initialize parameters ------------------

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.sequence_length], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # L2 regularization loss
        l2_loss = tf.constant(0.0, dtype=tf.float32)

# End initialize ------------------

# Embedding step ------------------

        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.name_scope('embedding'):
                with tf.device('/cpu:0'):
                    self.d_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0), name='d_embeddings')
                    self.embedding_chars = tf.nn.embedding_lookup(self.d_embeddings, self.x)
                    self.embedding_chars_expanded = tf.expand_dims(self.embedding_chars, -1)  # shape = [batch_size, sequence_length, emb_size, 1]

# End embedding step --------------

# Convolution + pooling + flatten + highway + fully_connected ----------------

            self.pooled_outputs = self.convolution_layer(self.filter_sizes, self.num_filters,
                                                    self.embed_size, self.embedding_chars_expanded, self.sequence_length)

            # Combine all the pooled features(Flatten step)
            num_filters_total = sum(self.num_filters)
            self.h_pool = tf.concat(self.pooled_outputs, 3)
            self.h_pool_flatten = tf.reshape(self.h_pool, [-1, num_filters_total])  # one channel for one neuron, shape = [batch_size, num_filters_total]

# End Convolution... step -----------

# Some Special Networks -------------

            # Add highway
            with tf.name_scope('highway'):
                self.d_highway = self.highway(self.h_pool_flatten, self.h_pool_flatten.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope('dropout'):
                self.d_dropout = tf.nn.dropout(self.d_highway, self.dropout_keep_prob)

            # Final scores and predictions(Fully connected)
            with tf.name_scope('output'):
                W = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.d_dropout, W, b, name="scores")
                self.y_pred_for_acu = tf.nn.softmax(self.scores)    # shape = [batch_size, 2]
                self.predictions = tf.argmax(self.scores, 1, name='predictions')

# End Spacial Networks -----------------------

# Config loss and optimizer & update parameters------

            # CalculateMean cross-entropy loss
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y))
                self.loss = loss + self.l2_reg_lambda * l2_loss     # L2_norm

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = self.d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)

        # weight clipping
        # gradients_list = [x for x, _ in grads_and_vars]
        # variable_list = [x for _, x in grads_and_vars]
        # gradients_list, _ = tf.clip_by_global_norm(gradients_list, 5.0)
        # grads_and_vars = [i for i in zip(gradients_list, variable_list)]

        self.d_updates = self.d_optimizer.apply_gradients(grads_and_vars)

        # WGAN Clip weights
        # self.d_clip = [p.assign(tf.clip_by_value(p, 1e-50, 10.0)) for p in list(self.params)]

# End Config & update model parameters --------------

    def convolution_layer(self, filter_sizes, num_filters, embed_size, embedding_chars_expanded, sequence_length):
        # Create a convolution + max-pooling layer for each filter size
        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            with tf.name_scope("conv_max-pooling_%s" % filter_size):
                # Convolution layer
                W = tf.Variable(tf.truncated_normal([filter_size, embed_size, 1, num_filter], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                conv = tf.nn.conv2d(
                    embedding_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                # Apply nonlinearity
                activation_output = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    activation_output,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],     # shape = [batch_size, height(seq_length), width(emb_size), channel]
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pooled"
                )
                pooled_outputs.append(pooled)

        return pooled_outputs

    def highway(self, input_, size, num_layers=1, bias=-2.0):
        output = 0.0
        with tf.variable_scope('Highway'):
            for idx in range(num_layers):
                g = tf.nn.relu(self.linear(input_, size, scope='highway_lin_%d' % idx))
                t = tf.sigmoid(self.linear(input_, size, scope='highway_gate_%d' % idx) + bias)
                output = t * g + (1.0 - t) * input_
                input_ = output

        return output

    def linear(self, input_, output_size, scope="SimpleLinear"):
        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]

        # Computation
        with tf.variable_scope(scope):
            matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
            bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

        return tf.matmul(input_, tf.transpose(matrix, [1, 0])) + bias_term

    def pretrain_forward(self, sess, x, y, dropout):
        outputs = sess.run([self.d_updates, self.loss], feed_dict={self.x: x, self.y: y, self.dropout_keep_prob: dropout})
        return outputs
