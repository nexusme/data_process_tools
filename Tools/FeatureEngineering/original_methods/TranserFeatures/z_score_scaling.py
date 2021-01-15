import pandas as pd


def z_score(df_name, col):
    """
    z-score编码
    :param col: 待处理列名
    :param df_name: dataframe
    """
    data = df_name[col]
    print(data)
    length = len(data)
    total = sum(data)
    ave = float(total) / length
    list_data = list(data)
    tmp_sum = sum([pow(list_data[i] - ave, 2) for i in range(length)])
    tmp_sum = pow(float(tmp_sum) / length, 0.5)
    for i in range(length):
        list_data[i] = (list_data[i] - ave) / tmp_sum
    df_name[col] = pd.DataFrame(list_data)
    return df_name
