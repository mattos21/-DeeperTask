import pandas as pd
import os

path = os.getcwd()

def make_dir():

        os.mkdir(path + '/train/upright')

        os.mkdir(path + '/train/rotated_left')

        os.mkdir(path + '/train/rotated_right')

        os.mkdir(path + '/train/upside_down')

make_dir()
train = pd.read_csv("train.truth.csv").values

for i in range(len(train)):
    if 'upright' in train[i][1]:
        os.rename(path + "/train/{}".format(train[i][0]), path + "/train/upright/{}".format(train[i][0]))

    elif 'rotated_left' in train[i][1]:
        os.rename(path + "/train/{}".format(train[i][0]), path + "/train/rotated_left/{}".format(train[i][0]))

    elif 'rotated_right' in train[i][1]:
        os.rename(path + "/train/{}".format(train[i][0]), path + "/train/rotated_right/{}".format(train[i][0]))

    elif 'upside_down' in train[i][1]:
        os.rename(path + "/train/{}".format(train[i][0]), path + "/train/upside_down/{}".format(train[i][0]))

