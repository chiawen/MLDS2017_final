import numpy as np
import collections
import _pickle as cPickle
import tensorflow as tf
from tensorflow.python.platform import gfile
import math
import sys
import re
try:
	import cPickle as pickle
except ImportError:
	import pickle

class Data(object):
	def __init__(self, train_data, train_label, test_data, test_label):
		self.current = 0

		self.train_data = train_data
		self.train_label = train_label

		self.test_data = test_data
		self.test_label = test_label

		self.length = self.train_data.shape[0]

	def next_batch(self, size):
		if self.current == 0:
			index = np.random.permutation(np.arange(self.length))
			self.train_data = self.train_data[index]
			self.train_label = self.train_label[index]

		if self.current + size < self.length:
			d, l = self.train_data[self.current:self.current+size], self.train_label[self.current:self.current+size]
			self.current += size
		else:
			d, l = self.train_data[self.current:], self.train_label[self.current:]
			self.current = 0

		return d, l


def gen_data(train_size=100000, test_size=10000, seq_length=150):

	size = train_size + test_size
	data = []

	train_rand = np.random.uniform(0, 1, [size, seq_length])
	train_c = np.zeros([size, seq_length])

	rand_idx = np.random.choice(seq_length, [size, 2])
	w_idx = np.equal(rand_idx[:, 0], rand_idx[:, 1])
	rand_idx[w_idx, 1] = (rand_idx[w_idx, 1] + 1) % seq_length


	train_c[np.arange(size)[:, None], rand_idx] = 1

	lable = np.sum(train_rand[np.arange(size)[:, None], rand_idx], axis=1)

	for i in range(size):
		data.append(np.concatenate(([train_rand[i]], [train_c[i]])).T)

	data = np.array(data)

	cPickle.dump((data[:train_size], lable[:train_size]), open("train.dat", 'wb'))
	cPickle.dump((data[train_size:], lable[train_size:]), open("test.dat", 'wb'))

	return data[:train_size], lable[:train_size], data[train_size:], lable[train_size:]








