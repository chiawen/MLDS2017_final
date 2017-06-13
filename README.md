MLDS Final
====
This is the <em>Machine Learning and having it deep and structured</em> course's final project experiment source code. The goal is to test the RNN parameters initializtation setting. You can see the complete experiment settings and content from [here](https://ntumlds.wordpress.com/2017/03/28/r05922027_沙拉和狗/). 

## Environment
python3 <br />
TensorFlow 1.0 <br />

## Experiments 

- Identity Recurrent Neural Network exp1 - adding problem: 
  - adding/ 
- LSTM forget gate bias: 
  - adding-lstm_bias/ 

## Data

- Identity Recurrent Neural Network exp1 - adding problem: <br />
  Use TensorFlow package to import the MNIST dataset.
- LSTM forget gate bias: <br />
  Source code will autogenerate a random training and testing data.


## Usage 
1. Go to the experiment folder that you want to test.
2. Start training !

## Train
IRNN exp1 - adding problem sequence length 50
```
$ cd adding/ 
$ python3 main.py --epochs 300 --seq_length 50 --lstm_lr 0.01 --rnn_tanh_lr 0.01 --rnn_relu_lr 0.01 --irnn_lr 0.01
```
IRNN exp1 - adding problem with sequence length 150
```
$ cd adding/ 
$ python3 main.py --epochs 300 --seq_length 150 --lstm_lr 1e-4 --rnn_tanh_lr 1e-4 --rnn_relu_lr 1e-4 --irnn_lr 1e-4
```
LSTM forget bais 
```
$ cd adding-lstm_bias/ 
$ python3 main.py --lstm_lr 1e-2
```

## Experiment Results
- Identity Recurrent Neural Network exp1 sequence length 50: <br />
  <img src="https://github.com/chiawen/MLDS2017_final/blob/master/asset/adding-1.png" width="512x">
- Identity Recurrent Neural Network exp1 sequence length 150: <br />
  <img src="https://github.com/chiawen/MLDS2017_final/blob/master/asset/adding-2.png" width="512x">
- LSTM forget gate bias: <br />
  <img src="https://github.com/chiawen/MLDS2017_final/blob/master/asset/forget_bias.png" width="512x">








