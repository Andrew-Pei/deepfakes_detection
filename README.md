# deepfakes_detection

## Function of each file：

train.py: the main program when you are training this model.

models.py: the details of the models( there are some unused models such as lstm, lrcn. Just ignore them.)

data.py: data process module.

## Requirements:
My code runs in this environment：ubuntu 16.04.6  nvidia-docker 19.03.4  tensorflow 2.1 keras 2.3.1 python 3.6
in other environments, I am not sure if it can run successfully

## Usage：

make sure you have download the dataset and change its location path to your own path in train.py
Then just run:
`python train.py`
