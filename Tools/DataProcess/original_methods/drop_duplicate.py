import pandas as pd
import numpy as np


def drop_duplicate(df, columns):
    """
    去重特定列中的重复值
    :param df:
    :param columns:
    :return:
    """
    df = df.drop_duplicates(subset=columns)
    return df


