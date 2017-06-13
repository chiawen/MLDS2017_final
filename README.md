MLDS Final Experiments
====
Machine Learning and having it deep and structured final project experiments' source code.


## Environment
python3 <br />
TensorFlow 1.0 <br />

## Experiments 

- LSTM forget gate bias: 
  - adding-lstm_bias/ 
- Identity Recurrent Neural Network exp1 - adding problem: 
  - adding/ 

## Data

- LSTM forget gate bias: <br />
  Source code will autogenerate a random training and testing data.
- Identity Recurrent Neural Network exp1 - adding problem: <br />
  Use TensorFlow package to import the MNIST dataset.


## Usage 
1. Go to the experiment folder that you want to test.
2. Start training !

## Train
cd to the chosen experiment folder.
Take IRNN exp1 for example
```
$ cd adding/ 
$ python3 main.py --epochs 300 --lstm_lr 0.01 --rnn_tanh_lr 0.01 --rnn_relu_lr 0.01 --irnn_lr 0.01
```

## Experiment Results
- LSTM forget gate bias: <br />
  <img src="https://github.com/chiawen/MLDS2017_final/blob/master/asset/forget_bias.png" width="512x">
- Identity Recurrent Neural Network exp1 sequence length 50: <br />
  <img src="https://github.com/chiawen/MLDS2017_final/blob/master/asset/adding-1.png" width="512x">
- Identity Recurrent Neural Network exp1 sequence length 150: <br />
  <img src="https://github.com/chiawen/MLDS2017_final/blob/master/asset/adding-2.png" width="512x">








