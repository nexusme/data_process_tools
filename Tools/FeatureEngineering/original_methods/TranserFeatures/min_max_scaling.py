from sklearn.preprocessing import MinMaxScaler


def min_max(df_name, col):
    """
    归一化
    :param col: 待处理列名
    :param df_name: dataframe
    """
    data = df_name[col]
    min_max_scaler = MinMaxScaler()
    min_max_param = min_max_scaler.fit_transform(data.values.reshape(-1, 1))
    df_name[col] = min_max_scaler.fit_transform(df_name[col].values.reshape(-1, 1), min_max_param)
    return df_name