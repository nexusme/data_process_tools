import pandas as pd
import numpy as np


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(df, col, target_col, min_samples_leaf=1, smoothing=1, noise_level=0.0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    df : dataframe
    col: column name waits for encoding
    target_col : target column name
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    series = df[col]
    target = df[target_col]
    assert len(series) == len(target)
    temp = pd.concat([series, target], axis=1)
    # Compute mean
    averages = temp.groupby(by=col)[target_col].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target_col] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to series
    ft_trn_series = pd.merge(
        series.to_frame(col),
        averages.reset_index().rename(columns={'index': target_col, target_col: 'average'}),
        on=col,
        how='left')['average'].rename(col + '_mean').fillna(prior)
    ft_trn_series.index = series.index
    df[col] = add_noise(ft_trn_series, noise_level)
    return df


# if __name__ == '__main__':
#     df_1 = pd.DataFrame([[5, 2, 1, 0],
#                          [7, 4, 6, 1],
#                          [3, 9, 6, 0],
#                          [1, 3, 1, 1]],
#                         columns=list('ABCy'))
#
#     df_test = pd.DataFrame([['m', 1],
#                             ['e', 1],
#                             ['s', 1],
#                             ['m', 1],
#                             ['m', 0],
#                             ['e', 0],
#                             ['e', 1]], columns=list('Ty'))
#
#     result = target_encode(df_test, col="T", target_col='y', min_samples_leaf=100, smoothing=10, noise_level=0.01)
#     print(result)
