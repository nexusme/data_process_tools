from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


def optimal_binning_boundary(df_name, col, target):
    """
    决策树分箱
    :param df_name: dataframe
    :param col: column name
    :param target: y
    :return:
    """
    boundary = []
    x = df_name[col].values
    y = df_name[target].values
    clf = DecisionTreeClassifier(criterion='entropy',
                                 max_leaf_nodes=6,
                                 min_samples_leaf=0.05)
    clf.fit(x.reshape(-1, 1), y)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:
            boundary.append(threshold[i])
    boundary.sort()
    boundary.insert(0, float('-inf'))
    boundary.append(float('inf'))
    return boundary


def decision_tree_feature_woe_iv(df_name, col, target):
    """
    决策树分箱 计算woe iv
    :param df_name: dataframe
    :param col: column which need to manage
    :param target: y
    :return:
    """
    boundary = optimal_binning_boundary(df_name, col, target)
    df1 = pd.concat([df_name[col], df_name[target]], axis=1)
    df1.columns = ['x', 'y']
    df1['bins'] = pd.cut(x=df_name[col], bins=boundary, right=False)
    grouped = df1.groupby('bins')['y']
    result_df = grouped.agg([('good', lambda y: (y == 0).sum()),
                             ('bad', lambda y: (y == 1).sum()),
                             ('total', 'count')])

    result_df['good_pct'] = result_df['good'] / result_df['good'].sum()
    result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()
    result_df['total_pct'] = result_df['total'] / result_df['total'].sum()

    result_df['bad_rate'] = result_df['bad'] / result_df['total']

    woe = np.log(result_df['good_pct'] / result_df['bad_pct'])
    result_df['woe'] = woe
    result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']
    iv = result_df['iv'].sum()
    print('result_df', result_df)
    print('boundary', boundary)
    print('woe', woe)
    print('iv', iv)
    return result_df, boundary, woe, iv
