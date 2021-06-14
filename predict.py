# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:58:31 2021

@author: vf199
"""

import pandas as pd
import numpy as np
from keras.models import load_model
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--model",default="model.h5",help="Input your model.")
parser.add_argument("--training",default="./data/sales_train.csv",help="Input your training data.")
parser.add_argument("--testing",default="./data/test.csv",help="Input your testing data.")
parser.add_argument("--output",default="submission2.csv",help="Output your result.")
args=parser.parse_args()

sales=pd.read_csv(args.training)
test=pd.read_csv(args.testing)

sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')
sales.tail(10)
dataset = sales.pivot_table(index=['item_id', 'shop_id'],values=['item_cnt_day'], columns='date_block_num', fill_value=0,aggfunc=np.sum)
dataset = dataset.reset_index()
dataset = pd.merge(test, dataset, on=['item_id', 'shop_id'], how='left')

X_test = np.expand_dims(dataset.values[:, 1:], axis=2)

model = load_model(args.model)

LSTM_prediction = model .predict(X_test[:,-33:,:])
LSTM_prediction = np.round(LSTM_prediction,2)
LSTM_prediction = LSTM_prediction.clip(0, 20)

submission = pd.DataFrame({'ID': test['ID'], 'item_cnt_month': LSTM_prediction.ravel()})
submission.to_csv(args.output,index=False)