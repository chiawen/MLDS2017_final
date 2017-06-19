import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

class MyRNNCell(RNNCell):
        def __init__(self, num_units, activation=tf.nn.relu, reuse=None):
                self._num_units = num_units
                self._activation = activation
                self._reuse = reuse
        
        @property
        def state_size(self):
                return self._num_units

        @property
        def output_size(self):
                return self._num_units

        def __call__(self, inputs, state, scope=None):
                with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
                        W_h = tf.get_variable("h_weights", shape=[self._num_units, self._num_units] ,initializer=tf.constant_initializer(np.identity(self._num_units)))#[self._num_units, self._num_units])#initializer=init)
                        W_i = tf.get_variable("i_weights", shape=[inputs.get_shape()[1], self._num_units])
                        b = tf.get_variable("bias", shape=[self._num_units], initializer=tf.constant_initializer(0))
                        output = self._activation(tf.matmul(inputs, W_i) + tf.matmul(state, W_h) + b)

                return output, output

class OrthRNNCell(RNNCell):
        def __init__(self, num_units, activation=tf.nn.relu, reuse=None):
                self._num_units = num_units
                self._activation = activation
                self._reuse = reuse
        
        @property
        def state_size(self):
                return self._num_units

        @property
        def output_size(self):
                return self._num_units

        def __call__(self, inputs, state, scope=None):
                with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
                        W_h = tf.get_variable("h_weights", shape=[self._num_units, self._num_units] ,initializer=tf.orthogonal_initializer(gain=1.0, dtype=tf.float32, seed=None))#[self._num_units, self._num_units])#initializer=init)
                        W_i = tf.get_variable("i_weights", shape=[inputs.get_shape()[1], self._num_units])
                        b = tf.get_variable("bias", shape=[self._num_units], initializer=tf.constant_initializer(0))
                        output = self._activation(tf.matmul(inputs, W_i) + tf.matmul(state, W_h) + b)

                return output, output



class LSTM(object):
        def __init__(self, seq_length, hidden_size, lr, forget_bias, activation, name):
                self.name = "LSTM + %s"%name
                with tf.variable_scope("lstm_%s"%name) as scope:
                        self.seq_in = tf.placeholder(tf.float32, [None, seq_length])
                        self.label = tf.placeholder(tf.float32, [None, 10])

                        seq_in = tf.image.resize_images(tf.reshape(self.seq_in, [-1, 28, 28, 1]), [14, 14])

                        seq_in = tf.reshape(seq_in, [-1, 196, 1])

                        print(seq_in)

                        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=forget_bias, state_is_tuple=True, activation=activation)
                        init_state = cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32)

                        outputs, states = tf.nn.dynamic_rnn(cell, 
                                                                                        seq_in,  
                                                                                        sequence_length=tf.fill([tf.shape(self.seq_in)[0]], 196), 
                                                                                        dtype=tf.float32, 
                                                                                        initial_state=init_state, 
                                                                                        scope="lstm_%s"%name)
                        f_outs = outputs[:,-1]

                        logit = tf.contrib.layers.fully_connected(
                                        f_outs, 10,
                                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                        activation_fn=None
                                        )
                        
                        self.loss = tf.contrib.losses.softmax_cross_entropy(logits=logit, onehot_labels=self.label)

                        p_argmax = tf.argmax(logit, axis=1)
                        l_argmax = tf.argmax(self.label, axis=1)
                        self.acc = tf.reduce_mean(tf.cast(tf.equal(p_argmax, l_argmax), tf.float32))

                optimizer = tf.train.GradientDescentOptimizer(lr)

                tvars = [var for var in tf.trainable_variables() if "lstm_%s"%name in var.name]

                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)

                self.updates = optimizer.apply_gradients(
                                                zip(grads, tvars))

class RNN(object):
        def __init__(self, seq_length, hidden_size, lr, activation, name):
                self.name = "RNN + %s"%name
                with tf.variable_scope("rnn_%s"%name) as scope:
                        self.seq_in = tf.placeholder(tf.float32, [None, seq_length])
                        self.label = tf.placeholder(tf.float32, [None, 10])

                        seq_in = tf.image.resize_images(tf.reshape(self.seq_in, [-1, 28, 28, 1]), [14, 14])

                        seq_in = tf.reshape(seq_in, [-1, 196, 1])

                        cell = tf.contrib.rnn.BasicRNNCell(hidden_size, activation=activation)
                        
                        init_state = cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32)

                        outputs, states = tf.nn.dynamic_rnn(cell, 
                                                                                        seq_in,  
                                                                                        sequence_length=tf.fill([tf.shape(self.seq_in)[0]], 196), 
                                                                                        dtype=tf.float32, 
                                                                                        initial_state=init_state, 
                                                                                        scope='rnn_%s' % name)
                        f_outs = outputs[:,-1]

                        logit = tf.contrib.layers.fully_connected(
                                        f_outs, 10,
                                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                        activation_fn=None
                                        )
                        
                        self.loss = tf.contrib.losses.softmax_cross_entropy(logits=logit, onehot_labels=self.label)

                        p_argmax = tf.argmax(logit, axis=1)
                        l_argmax = tf.argmax(self.label, axis=1)
                        self.acc = tf.reduce_mean(tf.cast(tf.equal(p_argmax, l_argmax), tf.float32))

                optimizer = tf.train.GradientDescentOptimizer(lr)

                tvars = [var for var in tf.trainable_variables() if "rnn_%s"%name in var.name]

                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)

                self.updates = optimizer.apply_gradients(
                                                zip(grads, tvars))

class IRNN(object):
        def __init__(self, seq_length, hidden_size, lr, activation, name):
                self.name = "IRNN"
                with tf.variable_scope("irnn_%s"%name) as scope:
                        self.seq_in = tf.placeholder(tf.float32, [None, seq_length])
                        self.label = tf.placeholder(tf.float32, [None, 10])

                        seq_in = tf.image.resize_images(tf.reshape(self.seq_in, [-1, 28, 28, 1]), [14, 14])

                        seq_in = tf.reshape(seq_in, [-1, 196, 1])

                        cell = MyRNNCell(hidden_size, activation=activation)
                        
                        init_state = cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32)

                        outputs, states = tf.nn.dynamic_rnn(cell, 
                                                                                        seq_in,  
                                                                                        sequence_length=tf.fill([tf.shape(self.seq_in)[0]], 196), 
                                                                                        dtype=tf.float32, 
                                                                                        initial_state=init_state, 
                                                                                        scope='irnn_%s' % name)
                        f_outs = outputs[:,-1]

                        logit = tf.contrib.layers.fully_connected(
                                        f_outs, 10,
                                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                        activation_fn=None
                                        )
                        
                        self.loss = tf.contrib.losses.softmax_cross_entropy(logits=logit, onehot_labels=self.label)

                        p_argmax = tf.argmax(logit, axis=1)
                        l_argmax = tf.argmax(self.label, axis=1)
                        self.acc = tf.reduce_mean(tf.cast(tf.equal(p_argmax, l_argmax), tf.float32))

                optimizer = tf.train.GradientDescentOptimizer(lr)

                tvars = [var for var in tf.trainable_variables() if "irnn_%s"%name in var.name]

                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)

                self.updates = optimizer.apply_gradients(
                                                zip(grads, tvars))

class ORNN(object):
        def __init__(self, seq_length, hidden_size, lr, activation, name):
                self.name = "Orthogonal RNN"
                with tf.variable_scope("ornn_%s"%name) as scope:
                        self.seq_in = tf.placeholder(tf.float32, [None, seq_length])
                        self.label = tf.placeholder(tf.float32, [None, 10])

                        seq_in = tf.image.resize_images(tf.reshape(self.seq_in, [-1, 28, 28, 1]), [14, 14])

                        seq_in = tf.reshape(seq_in, [-1, 196, 1])

                        cell = MyRNNCell(hidden_size, activation=activation)
                        
                        init_state = cell.zero_state(tf.shape(self.seq_in)[0], dtype=tf.float32)

                        outputs, states = tf.nn.dynamic_rnn(cell, 
                                                                                        seq_in,  
                                                                                        sequence_length=tf.fill([tf.shape(self.seq_in)[0]], 196), 
                                                                                        dtype=tf.float32, 
                                                                                        initial_state=init_state, 
                                                                                        scope='ornn_%s' % name)
                        f_outs = outputs[:,-1]

                        logit = tf.contrib.layers.fully_connected(
                                        f_outs, 10,
                                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                                        activation_fn=None
                                        )
                        
                        self.loss = tf.contrib.losses.softmax_cross_entropy(logits=logit, onehot_labels=self.label)

                        p_argmax = tf.argmax(logit, axis=1)
                        l_argmax = tf.argmax(self.label, axis=1)
                        self.acc = tf.reduce_mean(tf.cast(tf.equal(p_argmax, l_argmax), tf.float32))

                optimizer = tf.train.GradientDescentOptimizer(lr)

                tvars = [var for var in tf.trainable_variables() if "ornn_%s"%name in var.name]

                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 1)

                self.updates = optimizer.apply_gradients(
                                                zip(grads, tvars))

