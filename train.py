# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:39:33 2021

@author: vf199
"""

import numpy as np 
import pandas as pd 
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--training",default="./data/sales_train.csv",help="Input your training data.")
parser.add_argument("--testing",default="./data/test.csv",help="Input your testing data.")
parser.add_argument("--output",default="model.h5",help="Output your result.")
args=parser.parse_args()

# Import all of them 
sales=pd.read_csv(args.training)

# settings
import warnings
warnings.filterwarnings("ignore")

test=pd.read_csv(args.testing)

sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')
sales.tail(10)

dataset = sales.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0,aggfunc=np.sum)
dataset = dataset.reset_index()

dataset = pd.merge(test, dataset, on=['item_id', 'shop_id'], how='left')
dataset = dataset.fillna(0)

dataset = dataset.drop(['shop_id', 'item_id', 'ID'], axis=1)

#原始x_train出來會是二維 （214200, 33）,用np.expand_dims在第二維度後再多加一維,應該是因為lstm要三維data
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)

#第34個月（date_block_num)
y_train = dataset.values[:, -1:]

#x_train切完會是214200筆 時間序為33的資料

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model_kaggle_lstm = Sequential()
model_kaggle_lstm .add(LSTM(units=64, input_shape=(33,1)))
model_kaggle_lstm .add(Dropout(0.3))
model_kaggle_lstm .add(Dense(1))

model_kaggle_lstm .compile(loss='mse',
              optimizer='adam',
              metrics=['mean_squared_error'])

model_kaggle_lstm .summary()

from keras.callbacks import EarlyStopping

callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=3,mode='auto')]

history = model_kaggle_lstm .fit(X_train, y_train, batch_size=4096, epochs=30,callbacks=callbacks_list)
model_kaggle_lstm .save(args.output)
