# Predict_Future_Sales
## 摘要
本專案為Kaggle上[**Predict Future Sales**](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)以預測未來銷售量為目標的一項比賽。具體目標為使用過去各間商店的各個商品的每日銷售量預測各商店中各種商品的未來一個月銷售量。本專案使用決策樹（Decision tree）建構預測模型，並使用過去36個月的銷售數據進行訓練，並預測未來一個月的銷售量。    

## 開發環境
本專案於**Python 3.7.10**中進行開發，其開發過程中所使用的套件包如下表所示，並提供[**requirements.txt**](https://github.com/vf19961226/Predict_Future_Sales/blob/main/requiretments.txt)以方便安裝環境。

|Package|Version|
|:---:|:---:|
|Pandas|1.2.4
|Scikit-learn|0.24.2
|Joblib|1.0.1

## 資料
本專案所使用的資料皆由[**Predict Future Sales**](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)提供，詳情請參閱[**Predict Future Sales/Data**](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)。其中本專案會使用的資料為
[**sales_train.csv**](https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/8587/868304/compressed/sales_train.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1623915098&Signature=jalDxH2H1J4gF3eOGJKbwZrB9EakJszY6ebpg2DFe36eEBts0dxaxB6AYtLMb7IaNorjGO0fJgCp8%2Fdj0KCapWzHxSsidj4MU9nAvnHDTDkPrNeg490nEaMLIJDQCfdPQCUjJwHyMtu5eiil8kqKCtZRO1bNViSsSvAS9L%2BE4OOmTFSVHim3fyhGnOv%2FCS37ySnJjSq2fuWwJFfMig4aPljq0mYdAy6Sd03rhr5dptbFc8%2B9YOcxdUU4SLCl%2F8G4%2FVFBWymDa67GbJ7DkoLBoTpSA9jM6rSJ01yBz0850eX35C56BgW4utC%2FjSRDoYqFOejY5S1hhRgEZOH0mZ9e2A%3D%3D&response-content-disposition=attachment%3B+filename%3Dsales_train.csv.zip)、
[**test.csv**](https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/8587/868304/compressed/test.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1623915260&Signature=b%2FXa%2FEBy3eK5SFJ1NIxIdzmXaleEWBrs0kGTfDnJiIkU1rHyuTQi2ltM7T4xBP%2FUtPyXNlRVSuz7asnvRjwjfEsuelDKKOiDew81%2BpU0Qs3lzPWNE3rU%2FXlT698IbuFgjIocRLlJKlT%2FXGz8061VieJso95369F1SqFlabMFV%2BMS6IzQjovm2OmpBHYHYHFGa5NXVq%2BnD0Fo5OUgxS2Nq8%2FVF4g%2B41NvHhANqlvTvd%2FcrPrRH40w5C%2B91pC6NHShP07XfLPdMnfy%2FL4YMFs79%2BSAdqVB6iZY%2BBKEY9q9tGnDSyQIlOqs0Q9d92QeORmFy%2FBDB5m%2F87DDpi1CLR67lQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dtest.csv.zip)以及
[**sample_submission.csv**](https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/8587/868304/compressed/sample_submission.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1623915300&Signature=reY%2BASGjgpMCVyd7zU99UZdJqXD5CrjIeyZ54fyLHnys3kS6e98tUDHKbozhl63jDnvkrX9ZjrUgawPMDyfgykf1Bf62mBZuObwwa1VnZfs205%2B8a8ASXup7lOGYU73dxxUepgjFmiW8vKDW8Ofw9Y30q%2F0zT1d0q10pLWcasIXRoo6uMAzA2tGgPy9eydkvEwSeproJ8oVYbPvPcJ8cLM3Xv1ozxgm4bdwcpHev7sHOq6qYNuz8iT5Bn31vrv9FtGVlrlm6azRR8iyGYkW3tXMgzr%2BSmxgUvg%2Bg1O3tPYjtSUGud4k5T%2By9gJJW%2FczjAnfXQQONyDgefGAEscebKA%3D%3D&response-content-disposition=attachment%3B+filename%3Dsample_submission.csv.zip)。

### 資料處理
#### sales_train.csv
1. 將時間格式由日-月-年改成年-月
