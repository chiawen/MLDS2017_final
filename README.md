MLDS Final Experiment 
====
Machine Learning and having it deep and structured final project experiments' source code.


## Environment
python3 <br />
tensorflow 1.0 <br />
scipy <br />

## Experiments

- LSTM forget gate bias setting: adding-lstm_bias/ <br />
- Identity Recurrent Neural Network exp 1 - adding problem: adding/ <br />

## Data

- LSTM forget gate bias: asdfasd 


## Usage 
1. Download hw3 data from data link, place the MLDS_HW3_dataset/ in the same directory and unzip the face.zip in MLDS_HW3_dataset/
2. Replace the tags in MLDS_HW3_dataset/sample_testing_text.txt to the right format. 
3. Start training !

## Train
First time use, you need to do the preprocessing
```
$ python3 main.py --prepro 1
```
If you already have done the preprocessing
```
$ python3 main.py --prepro 0
```
## Model
- dcgan structure
- use one hot encoding for condition tags

## Test 
This code will automatically dump the results for the tags specified in MLDS_HW3_dataset/sample_testing_text.txt every <em>dump_every</em> batches to the test_img/ folder. <br />

## Experiment Results
- blue hair blue eyes <br />
  ![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/0.jpg)
- gray hair green eyes <br />
  ![image](https://github.com/m516825/Conditional-GAN/blob/master/asset/1.jpg)








