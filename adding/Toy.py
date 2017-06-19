import tensorflow as tf
import time 
import numpy as np
import os
from model import LSTM, RNN, IRNN, ORNN
import progressbar as pb
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Toy(object):
    def __init__(self, data, FLAGS):
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.data = data
        self.FLAGS = FLAGS
        self.models = {}
        self.scores = {}
        self.gen_path()

    def gen_path(self):
        # Output directory for models and summaries
        timestamp = str(time.strftime('%b-%d-%Y-%H-%M-%S'))
        self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
        print ("Writing to {}\n".format(self.out_dir))
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def build_model(self, output_graph=True):
        """
        self.lstm_relu = LSTM(seq_length=self.FLAGS.seq_length, 
                    hidden_size=self.FLAGS.hidden_size,
                    lr=self.FLAGS.lstm_lr,
                    activation=tf.nn.relu,
                    name="relu")
        """
        self.lstm_bias1 = LSTM(seq_length=self.FLAGS.seq_length, 
                    hidden_size=self.FLAGS.hidden_size,
                    lr=self.FLAGS.lstm_lr,
                    activation=tf.nn.tanh,
                    forget_bias=1.0,
                    name="bias1")
        self.lstm_bias5 = LSTM(seq_length=self.FLAGS.seq_length, 
                    hidden_size=self.FLAGS.hidden_size,
                    lr=self.FLAGS.lstm_lr,
                    activation=tf.nn.tanh,
                    forget_bias=5.0,
                    name="bias5")
        self.rnn_tanh = RNN(seq_length=self.FLAGS.seq_length, 
                    hidden_size=self.FLAGS.hidden_size,
                    lr=self.FLAGS.rnn_tanh_lr,
                    activation=tf.nn.tanh,
                    name="tanh")
        self.rnn_relu = RNN(seq_length=self.FLAGS.seq_length, 
                    hidden_size=self.FLAGS.hidden_size,
                    lr=self.FLAGS.rnn_relu_lr,
                    activation=tf.nn.relu,
                    name="relu")
        self.irnn_relu = IRNN(seq_length=self.FLAGS.seq_length, 
                    hidden_size=self.FLAGS.hidden_size,
                    lr=self.FLAGS.irnn_lr,
                    activation=tf.nn.relu,
                    name="")
        self.ornn = ORNN(seq_length=self.FLAGS.seq_length, 
                    hidden_size=self.FLAGS.hidden_size,
                    lr=self.FLAGS.ornn_lr,
                    activation=tf.nn.tanh,
                    name="")
        """
        self.irnn_tanh = IRNN(seq_length=self.FLAGS.seq_length, 
                    hidden_size=self.FLAGS.hidden_size,
                    lr=self.FLAGS.irnn_lr,
                    activation=tf.nn.tanh,
                    name="tanh")
        """
        #self.models[self.lstm_relu.name] = self.lstm_relu
        self.models[self.lstm_bias1.name] = self.lstm_bias1
        self.models[self.lstm_bias5.name] = self.lstm_bias5
        self.models[self.rnn_tanh.name] = self.rnn_tanh
        self.models[self.rnn_relu.name] = self.rnn_relu
        self.models[self.irnn_relu.name] = self.irnn_relu
        self.models[self.ornn.name] = self.ornn
        #self.models[self.irnn_tanh.name] = self.irnn_tanh

        self.sess.run(tf.global_variables_initializer())

        # self.sess.run([self.irnn.w_assign, self.irnn.b_assign])

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def train(self):
        batch_num = self.data.length//self.FLAGS.batch_size if self.data.length%self.FLAGS.batch_size==0 else self.data.length//self.FLAGS.batch_size + 1
        result = 0.

        for k, v in self.models.items():
            self.scores[k] = []

        for ep in range(self.FLAGS.epochs):
            print("Epochs {}".format(ep))
            costs = {}
            for k, v in self.models.items():
                costs[k] = 0. 
            pbar = pb.ProgressBar(widget=[pb.Percentage(), pb.Bar, pb.ETA()], maxval=batch_num).start()
            for b in range(batch_num):
                d_d, l_d = self.data.next_batch(self.FLAGS.batch_size)

                for name, model in self.models.items():

                    feed_dict = {
                        model.seq_in:d_d,
                        model.label:l_d
                    }
                    loss, _ = self.sess.run([model.loss, model.updates], feed_dict=feed_dict)
                    costs[name] = loss / batch_num
                pbar.update(b+1)
            pbar.finish()
            


            for name, cost in sorted(costs.items(), key=lambda x:x[0]):
                print(">>{} cost: {}".format(name, cost))
            print("---------------------------------")
            for name, model in sorted(self.models.items(), key=lambda x:x[0]):
                score = self.eval(model)
                self.scores[name].append(self.eval(model))
                print(">>{} MSE: {}".format(name, score))
            self.dump_curve()

    def dump_curve(self):
        handles = []
        colors = ['r', 'b', 'g', 'm', 'k']
        for idx, (name, scores) in enumerate(sorted(self.scores.items(), key=lambda x:x[0])):
            handles.append(plt.plot(scores, colors[idx], label=name)[0])
        plt.legend(handles=handles)
        plt.xlabel('Epochs')
        plt.ylabel('Test MSE')
        plt.ylim(0,0.8,0.1)
        plt.title('Adding two numbers in a sequence of {} numbers'.format(self.FLAGS.seq_length))
        plt.savefig('test.png')

    def eval(self, model):

        feed_dict = {
            model.seq_in:self.data.test_data,
            model.label:self.data.test_label
        }

        MSE = self.sess.run(model.loss , feed_dict=feed_dict)

        return MSE

