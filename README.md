## CCKS_2019_Task3

### Task

[Inter-Personal Relationship Extraction](https://github.com/guangxush/CCKS2019_Task3/blob/master/documents/CCKS2019-eval-task3.pdf)

### Competition

[ccks_2019_ipre](https://www.biendata.com/competition/ccks_2019_ipre/)

### Environment
- keras
- python3
- sklearn
- pandas
- numpy
- h5py
- xgboost


### DataSet

|data set|Classification Task|Classes|Train|Dev|Test|Date|
|:---:|:---:|:----:|:--:|:--:|:--:|:--:|
|A|Relationship Extraction|35|287351|38416|77092|2019-07-25|

train classes analysis

| 9 | 0      | 18  | 10   | 16   | 11   | 22 | 32   | 27 | 20 | 8  | 15 | 31   | 33   | 12   | 31 | 5   | 4    | 10 | 25 | 31 | 7   | 17  | 19  | 30   | 1    | 28 | 21 | 29  | 13  | 6  | 3   | 24 | 34  | 23  | 14 | 2   | 26  |
|---|--------|-----|------|------|------|----|------|----|----|----|----|------|------|------|----|-----|------|----|----|----|-----|-----|-----|------|------|----|----|-----|-----|----|-----|----|-----|-----|----|-----|-----|
| 9 | 248850 | 532 | 6859 | 1673 | 1383 | 22 | 1266 | 67 | 77 | 40 | 19 | 1263 | 2900 | 2627 | 4  | 245 | 5513 | 33 | 13 | 1  | 291 | 637 | 805 | 1610 | 8135 | 24 | 77 | 165 | 830 | 69 | 183 | 30 | 547 | 158 | 46 | 218 | 119 |

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


|model|precision|recall|f1|auc|accuracy|A Test|
|---------|:---:|:----:|:--:|:--:|:--:|:--:|
|cnn_base_base|0.9622|0.9622|0.1778|0.5928|0.9622|0.17811|
|cnn_base_limit_dis|0.9614|0.9614|0.1793|0.5946|0.9614|0.16864|
|bilstm_base|0.9427|0.9427|0.1890|0.6932|0.9427|0.18867|
|cnn_multi|0.9435|0.9435|0.1417|0.6770|0.9435|0.12147|
|bilstm_multi|0.9577|0.9577|0.1758|0.6102|0.9577|0.19007|
|cnn_base_multi_300|0.9655|0.9655|0.1862|0.5918|0.9655|0.19073|
|cnn_base|0.9565|0.9565|0.2217|0.6459|0.9565|0.22301|


