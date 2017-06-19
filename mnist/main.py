import tensorflow as tf
import numpy as np
import os
import data_utils
import _pickle as cPickle
from data_utils import Data
from Toy import Toy

tf.flags.DEFINE_integer("hidden_size", 100, "RNN hidden size")
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs", 300, "Number of training epochs")
tf.flags.DEFINE_integer("dev_size", 1, "validation size")
tf.flags.DEFINE_float("lstm_lr", 8e-2, "training learning rate")
tf.flags.DEFINE_float("rnn_tanh_lr", 5e-3, "training learning rate")
tf.flags.DEFINE_float("rnn_relu_lr", 5e-3, "training learning rate")
tf.flags.DEFINE_float("irnn_lr", 5e-3, "training learning rate")
tf.flags.DEFINE_integer("seq_length", 50, "validation size")

tf.flags.DEFINE_boolean("prepro", True, "preprocess word to vector matrix")
tf.flags.DEFINE_boolean("eval", False, "evaluation")

tf.flags.DEFINE_string("train_data", "./chat.txt", "Training data path")
tf.flags.DEFINE_string("vocab", "./prepro/vocab.dat", "vocab processor")
tf.flags.DEFINE_string("prepro_train", "./prepro/train.dat", "preprocessed training data")
tf.flags.DEFINE_string("prepro_dir", "./prepro/", "")
tf.flags.DEFINE_string("output", "pred.csv", "")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def main(_):

        for k, v in FLAGS.__flags.items():
                print("{}={}".format(k, v))

        if not os.path.exists(FLAGS.prepro_dir):
                os.makedirs(FLAGS.prepro_dir)

        model = Toy(FLAGS)

        model.build_model()

        model.train()

if __name__ == '__main__':
        tf.app.run()

