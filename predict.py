# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:51:41 2021

@author: vf19961226
"""


import pandas as pd
import joblib
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--model",default="model.pkl",help="Input your model.")
parser.add_argument("--training",default="./data/sales_train.csv",help="Input your training data.")
parser.add_argument("--testing",default="./data/test.csv",help="Input your testing data.")
parser.add_argument("--output",default="submission.csv",help="Output your result.")
args=parser.parse_args()

sales_data = pd.read_csv(args.training)
#date 的 datatype從object to datatime64
sales_data['date']=pd.to_datetime(sales_data['date'])
#sales_data.head(4)

#時間年,月,日改成年,月
sales_data['date']=sales_data['date'].dt.strftime('%Y-%m')
#按照日期大小排列
sales_data.head().sort_values(by='date')

#丟掉'date_block_num','item_price' 
sales_data.drop(['date_block_num','item_price'] , axis =1, inplace= True)
sales_data.head().sort_values(by='date')

sales_data=sales_data.groupby(['date','shop_id','item_id']).sum()
#sales_data.head()

#以['shop_id','item_id']為索引 column[date]為時間軸紀錄item_cnt_day
sales_data = sales_data.pivot_table(index=['shop_id','item_id'], columns='date', values='item_cnt_day', fill_value=0)
sales_data.head(10)

#插入索引
sales_data.reset_index(inplace=True)

test=pd.read_csv(args.testing)
df_for_test = pd.merge(test , sales_data , on = ['shop_id', 'item_id'], how = 'left')
df_for_test.drop(['ID', '2013-01'], axis =1, inplace=True)
df_for_test= df_for_test .fillna(0)
X_test = df_for_test

model_for_predict = joblib.load(args.model)
prediction = model_for_predict.predict(X_test)
print(prediction.shape)
print(prediction[:10])

#四捨五入 有做四捨五入效果比沒做的更趨近答案0.1
prediction = list(map(round, prediction))

df_submission = pd.read_csv('./data/sample_submission.csv')
print(df_submission.shape)
df_submission.head()

df_submission['item_cnt_month'] = prediction
df_submission.to_csv(args.output, index=False)
df_submission.head()