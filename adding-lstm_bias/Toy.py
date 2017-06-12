import tensorflow as tf
import time 
import numpy as np
import os
from model import LSTM, RNN, IRNN
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
		self.lstm_0 = LSTM(seq_length=self.FLAGS.seq_length, 
					hidden_size=self.FLAGS.hidden_size,
					lr=self.FLAGS.lstm_lr,
					activation=tf.nn.tanh,
					forget_bias=0.0,
					name="forget_bias_0.0")
		self.lstm_1 = LSTM(seq_length=self.FLAGS.seq_length, 
					hidden_size=self.FLAGS.hidden_size,
					lr=self.FLAGS.lstm_lr,
					activation=tf.nn.tanh,
					forget_bias=1.0,
					name="forget_bias_1.0")
		self.lstm_5 = LSTM(seq_length=self.FLAGS.seq_length, 
					hidden_size=self.FLAGS.hidden_size,
					lr=self.FLAGS.lstm_lr,
					activation=tf.nn.tanh,
					forget_bias=5.0,
					name="forget_bias_5.0")
		self.lstm_10 = LSTM(seq_length=self.FLAGS.seq_length, 
					hidden_size=self.FLAGS.hidden_size,
					lr=self.FLAGS.lstm_lr,
					activation=tf.nn.tanh,
					forget_bias=10.0,
					name="forget_bias_10.0")

		self.models[self.lstm_0.name] = self.lstm_0
		self.models[self.lstm_1.name] = self.lstm_1
		self.models[self.lstm_5.name] = self.lstm_5
		self.models[self.lstm_10.name] = self.lstm_10

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
					costs[name] += loss / batch_num
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
		colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k']
		for idx, (name, scores) in enumerate(sorted(self.scores.items(), key=lambda x:x[0])):
			handles.append(plt.plot(scores, colors[idx], label=name)[0])
		plt.legend(handles=handles)
		plt.xlabel('Epochs')
		plt.ylabel('Test MSE')
		plt.ylim(0,0.5,0.1)
		plt.title('Adding two numbers in a sequence of {} numbers'.format(self.FLAGS.seq_length))
		plt.savefig('test.png')

	def eval(self, model):

		feed_dict = {
			model.seq_in:self.data.test_data,
			model.label:self.data.test_label
		}

		MSE = self.sess.run(model.loss , feed_dict=feed_dict)

		return MSE

