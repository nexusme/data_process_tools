from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def rf_set_missing_part(df, columns):
    """
    随机森林填补缺失值
    :param df:  dataframe
    :param columns:  column to fill
    :return:
    """
    # 将需要处理的放到首列
    df_m = df
    df_move = df_m[columns]
    df_m = df_m.drop(columns, axis=1)
    df_m.insert(0, columns, df_move)

    # 把数值型特征放入随机森林里
    missing_part_df = df_m

    known_part = missing_part_df[missing_part_df[columns].notnull()].fillna(0).values

    unknown_part = missing_part_df[missing_part_df[columns].isnull()].fillna(0).values
    y = known_part[:, 0]  # y 第一列数据
    x = known_part[:, 1:]  # x 是特征属性值，后面几列
    rfr = RandomForestRegressor(random_state=0, n_estimators=15, max_depth=10, n_jobs=-1)
    # 根据已有数据去拟合随机森林模型
    rfr.fit(x, y)
    # 预测缺失值
    predicted_results = rfr.predict(unknown_part[:, 1:])
    predicted_results = [int(x) for x in predicted_results]
    # print(predicted_results)
    # 填补缺失值
    df.loc[(df[columns].isnull()), columns] = predicted_results

    # print('Missing data has been filled up.')
    print(df)
    return df


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[np.nan, 2, np.nan, 0],
#                          [3, 4, 1, 1],
#                          [2, np.nan, 2, 5],
#                          [np.nan, 3, np.nan, 4]],
#                         columns=list('ABCD'))
#
#     # print(df_1)
#     d = rf_set_missing_part(df_1, 'B')
#     # print(d)
