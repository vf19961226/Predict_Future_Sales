# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:36:50 2021

@author: vf19961226
"""


import pandas as pd
import warnings
warnings.filterwarnings('ignore') 
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--training",default="./data/sales_train.csv",help="Input your testing data.")
parser.add_argument("--output",default="model.pkl",help="Output your result.")
args=parser.parse_args()

#匯入檔案
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
#sales_data.head()

#試畫df_for_test row0 的散點圖
'''
select_row_0 = df_for_test.loc[0] #資料型別pandas.core.series.Series
select_row_0 =select_row_0.to_frame().T #轉換資料型別to dataframe
select_row_0.drop(['shop_id','item_id'], axis =1, inplace=True)
month = select_row_0.columns
row_0_data = select_row_0.values 
plt.figure(figsize=(10, 3), dpi=100)
plt.scatter(month, row_0_data)
plt.show()  
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

Y_train = sales_data['2015-10'].values
X_train = sales_data.drop(['2015-10'], axis = 1)

x_train, x_val, y_train, y_val = train_test_split( X_train, Y_train, test_size=0.2, random_state=101)

model_randomforest = RandomForestRegressor(  n_estimators=100,
                                             criterion='mse',
                                             max_depth=None,
                                             min_samples_split=2,
                                             min_samples_leaf=1,
                                             min_weight_fraction_leaf=0.0,
                                             max_features='auto',
                                             max_leaf_nodes=None,
                                             min_impurity_split=1e-07,
                                             bootstrap=True,
                                             oob_score=False,
                                             n_jobs=1,
                                             random_state=None,
                                             verbose=0,
                                             warm_start=False)
model_randomforest.fit(x_train,y_train)


print('Train loss:', mean_squared_error(y_train, model_randomforest.predict(x_train)))
print('Test loss:', mean_squared_error(y_val, model_randomforest.predict(x_val)))
print('Test R-squared :', model_randomforest.score(x_train,y_train))
#儲存model (我不知道為什麼不能用model.save) 這是找到的替代方案
joblib.dump(model_randomforest, args.output) 