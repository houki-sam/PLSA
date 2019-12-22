import numpy as np
import pandas as pd

#DataFrameを各列ごとにone-hot表現で変換する
def one_hot(DataFrame):
    
    #収納list
    onehot_list = []
    columns_list = []

    #listに変換し、各列ごとにone hot vectorを作成
    for row in DataFrame.columns:
        stack_one_hot , stack_columns = create(DataFrame[row])
        #リストに収納
        onehot_list.append(stack_one_hot)
        columns_list.append(stack_columns)

    return onehot_list, columns_list


#数え上げ
def create(series):
    #ユニークな変数を検索
    unique_list = list(set(series))
    #各数字に対応した辞書を作成
    columns = {key:i for i,key in enumerate(unique_list)}
    #逆引き辞書を作成
    reverse_columns = {v:k for k, v in columns.items()}
    #columnsに基づき置換
    replace_series = series.replace(columns).tolist()

    #one hotへ変換
    one_hot = np.identity(len(columns))[replace_series]
    #ラベルを作成
    label = [reverse_columns[x] for x in range(len(reverse_columns))]
    return one_hot, label