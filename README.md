# Predict_Future_Sales
## 摘要
本專案為Kaggle上[**Predict Future Sales**](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)以預測未來銷售量為目標的一項比賽。具體目標為使用過去各間商店的各個商品的每日銷售量預測各商店中各種商品的未來一個月銷售量。本專案使用隨機森林（Random Forest）建構預測模型，並使用過去35個月的銷售數據進行訓練，並預測未來一個月的銷售量。    

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
#### 訓練資料
1. 匯入sales_train.csv與test.csv
2. 將sales_train.csv中的時間格式由日-月-年改成年-月，並將日期由遠到近排列
3. 將sales_train.csv中用不到的資訊刪除，我們刪除了**date_block_num**和**item_price**
4. 計算每間商店各項商品的每月銷量
5. 匯出**test.csv**中要預測的商店與品項
6. 將資料以8比2的比例切割成訓練集與驗證集

#### 測試資料
1. 匯入sales_train.csv與test.csv
2. 將sales_train.csv中的時間格式由日-月-年改成年-月，並將日期由遠到近排列
3. 將sales_train.csv中用不到的資訊刪除，我們刪除了**date_block_num**和**item_price**
4. 計算每間商店各項商品的每月銷量
5. 匯出**test.csv**中要預測的商店與品項
6. 刪除最早一個月的銷售資料

### 模型架構
本專案之訓練模型使用Scikit-learn中的隨機森林（Random Forest）建構，其參數設定如下表所示。

|Name|Parameter|
|:---|:---|
|n_estimators|100
|criterion|mse
|max_depth|None
|min_samples_split|2
|min_samples_leaf|1
|min_weight_fraction_leaf|0.0
|max_features|auto
|max_leaf_nodes|None
|min_impurity_split|1e-07
|bootstrap|True
|oob_score|False
|n_jobs|1
|random_state|None
|verbose|0
|warm_start|False

### 如何使用
#### 準備資料
因某些資料過於龐大無法上傳至Github，如
[**sales_train.csv**](https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/8587/868304/compressed/sales_train.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1623915098&Signature=jalDxH2H1J4gF3eOGJKbwZrB9EakJszY6ebpg2DFe36eEBts0dxaxB6AYtLMb7IaNorjGO0fJgCp8%2Fdj0KCapWzHxSsidj4MU9nAvnHDTDkPrNeg490nEaMLIJDQCfdPQCUjJwHyMtu5eiil8kqKCtZRO1bNViSsSvAS9L%2BE4OOmTFSVHim3fyhGnOv%2FCS37ySnJjSq2fuWwJFfMig4aPljq0mYdAy6Sd03rhr5dptbFc8%2B9YOcxdUU4SLCl%2F8G4%2FVFBWymDa67GbJ7DkoLBoTpSA9jM6rSJ01yBz0850eX35C56BgW4utC%2FjSRDoYqFOejY5S1hhRgEZOH0mZ9e2A%3D%3D&response-content-disposition=attachment%3B+filename%3Dsales_train.csv.zip)以及訓練好的模型[**model.pkl**](https://drive.google.com/file/d/1-xPgGrvuaxkkjLejsiisrdr57rJbPhrQ/view?usp=sharing)
，**sales_train.csv**請自行由[**Predict Future Sales/Data**](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)下載並放置於[**data資料夾**](https://github.com/vf19961226/Predict_Future_Sales/tree/main/data)中，另外**model.pkl**部分[**點此處下載**](https://drive.google.com/file/d/1-xPgGrvuaxkkjLejsiisrdr57rJbPhrQ/view?usp=sharing)並放置於程式[**根目錄**](https://github.com/vf19961226/Predict_Future_Sales)中，其餘較小的資料均以上傳於[**data資料夾**](https://github.com/vf19961226/Predict_Future_Sales/tree/main/data)中。

#### 訓練模型
使用[**train.py**](https://github.com/vf19961226/Predict_Future_Sales/blob/main/train.py)進行訓練，本程式中包含了訓練資料的處理以及訓練模型兩部分。可使用以下指令於命令視窗中執行，其中命令參數說明如下表所示。

    python train.py --training "your training data(**.csv)" --output "your output model(**.pkl)"

|Name|Input|Default
|:---:|---|---
|--training|訓練資料|./data/sales_train.csv
|--output|輸出訓練模型|model.pkl

#### 測試模型
使用[**predict.py**](https://github.com/vf19961226/Predict_Future_Sales/blob/main/predict.py)進行測試，本程式中包含了測試資料的處理以及使用模型預測結果輸出兩部分。可使用以下指令於命令視窗中執行，其中命令參數說明如下表所示。

    python predict.py --model "your model(**.pkl)" --training "your training data(**.csv)" --testing "your testing data(**.csv)" --output "your output file(**.csv)"
    
|Name|Input|Default
|:---:|---|---
|--model|訓練完成的模型|model.pkl
|--training|訓練模型時使用的資料|./data/sales_train.csv
|--testing|測試目標商店與商品的列表|./data/test.csv
|--output|輸出預測結果|submission.csv

### 成果
#### 模型
本專案之模型訓練成果參數如下表所示。

|Item|Value|
|:---|:---|
|Train set mse|2.3564
|Test set mse|2.2870
|Test set score|0.8693

#### 預測
本專案使用[**test.csv**](https://github.com/vf19961226/Predict_Future_Sales/blob/main/data/test.csv)經過資料處理後搭配訓練好的模型[**model.pkl**](https://drive.google.com/file/d/1-xPgGrvuaxkkjLejsiisrdr57rJbPhrQ/view?usp=sharing)進行預測之結果紀錄於[**submission.csv**](https://github.com/vf19961226/Predict_Future_Sales/blob/main/submission.csv)之中。其格式如下表所示

|ID|item_cnt_month|
|:---|:---|
|0|0
|1|1
|2|0
|3|0
|4|1

#### 其他
本專案實作過程中更詳細的紀錄報告：[**report link**](https://docs.google.com/presentation/d/1sXl0K8NPNc3zbt92C4WY9i8vZh9Qkb9OhqChX9-LsSI/edit?usp=sharing)
