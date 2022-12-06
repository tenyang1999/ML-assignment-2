# ML-assignment-2
- 機器學習作業二：Naive Bayes與Random forest實作
- 使用Car Insurance Claim Prediction dataset進行預測
- 其中更使用sklearn的Random forest classifier 與 open source的xgboost去做比較
- 最終透過f1-score去評分，比較模型效能

## 功能

- 能夠透過在Classification_Task.ipynb檔中的classifier去簡易執行分類預測。

## 安裝
- 本次執行上會需要用到的package

```python
pip install numpy
pip install pandas
pip install sklearn
pip install xgboost
```
- naive_bayes.py，為單純貝式分類器，於Classification_Task.ipynb直接import即可套用
- random_forest.py，為隨機森林分類器，於Classification_Task.ipynb直接import即可套用
- tree.py，為決策樹分類器，可以直接取用於做分類任務，本次被引入random_forest.py，作為隨機森林的基底

## 使用方法
- tidy_train.csv為已經前處理過後的train.csv，詳細步驟寫於assignment.pdf中
- 執行Classification_Task.ipynb即可，將Problem1的classifier與Problem2的Cross-validation皆寫入同一個檔案中
