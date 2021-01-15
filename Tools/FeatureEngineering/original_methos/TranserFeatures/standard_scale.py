from sklearn.preprocessing import StandardScaler


def standard_scale(df_name, col):
    """
    标准化
    :param col: 待处理列名
    :param df_name: dataframe
    """
    data = df_name[col]
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scale_param = scaler.fit(data.values.reshape(-1, 1))
    df_name[col] = scaler.fit_transform(df_name[col].values.reshape(-1, 1), scale_param)
    return df_name