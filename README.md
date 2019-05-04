## CCKS_2019_Task3

### Task

[Inter-Personal Relationship Extraction](https://github.com/guangxush/CCKS2019_Task3/blob/master/documents/CCKS2019-eval-task3.pdf)


### Environment
- keras
- python3
- sklearn
- pandas
- numpy
- h5py
- xgboost


### DataSet

|data set |Classification Task|Classes|Train|Test|Date|
|---------|:---:|:----:|:--:|:--:|:--:|
|Train A|Sentiment analysis|2|25000|25000|25000*0.7|

### Data Statistics

|Class A|Class B|
|:---:|:---:|
|50%|50%|

### Use

- data_process:
```
    python data_process.py  :  generate model training data sets and vocabulary

    python train_word2vec.py : train word to vector

```

- model_train:
```
    python main.py

```

### Result

- save result in the 'result' folder

|solution|precision|recall|f1|accuracy|auc|time|
|---------|:---:|:----:|:--:|:--:|:--:|:--:|
|lstm+cnn|0.821617|0.821600|0.821598|0.821600|0.821600|00:16:43|

