import pandas as pd
import numpy as np


def cut_data(df_name, col, target):
    """
    等频分箱
    :param df_name: dataframe
    :param col: column
    :param target: y
    :return:
    """
    n = 10
    badsum = df_name[target].sum()
    goodsum = df_name[target].count() - badsum
    d1 = pd.concat((df_name[col], df_name[target]), axis=1)
    d1['bucket'] = pd.cut(df_name[col], n)
    d2 = d1.groupby('bucket', as_index=True)
    d3 = pd.DataFrame()
    d3['total'] = d2.count()[target]
    d3['max'] = d2.max()[col]
    d3['badsum'] = d2.sum()[target]
    d3['goodsum'] = d2.count()[target] - d3.badsum
    d3['bad_rate'] = d3['badsum'] / d3['total']
    d3['good_rate'] = d3['goodsum'] / d3['total']
    d3['total_rate'] = d3['total'] / df_name[target].count()
    woe = np.log((d3.goodsum / goodsum) / (d3.badsum / badsum))
    d3['woe'] = woe
    d3['iv'] = ((d3['goodsum'] / goodsum) - (d3['badsum'] / badsum)) * d3['woe']
    d3['total_iv'] = d3['iv'].sum()
    IV = d3['iv'].sum()
    d3['name'] = col
    cut = list(d3['max'].round(6))
    cut.insert(0, float('-inf'))
    cut[-1] = float('inf')
    print(d3
          )
    return d3, cut, IV, woe
