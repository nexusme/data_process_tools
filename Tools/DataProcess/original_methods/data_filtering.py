def data_filtering(df, columns, operation, value):
    if operation == 'dropna':
        data = df[columns].dropna()

    elif operation == '==':
        data = df[df[columns] == value]

    elif operation == '>=':
        data = df[df[columns] >= value]

    elif operation == '>':
        data = df[df[columns] > value]

    elif operation == '<=':
        data = df[df[columns] <= value]

    elif operation == '<':
        data = df[df[columns] < value]
    elif operation == '!=':
        data = df[df[columns] != value]

    return data
