def covert_columns_lower(df):
    """
    将列名全都变成小写
    """
    clean_column_name = []
    columns = df.columns
    for i in range(len(columns)):
        clean_column_name.append(columns[i].lower())
    df.columns = clean_column_name
    return df
