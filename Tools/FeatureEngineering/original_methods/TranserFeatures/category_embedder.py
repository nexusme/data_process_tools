import categorical_embedder as ce
import pandas as pd
from sklearn.model_selection import train_test_split

"""
    !pip install tensorflow_addons==0.8.3
    !pip install tqdm==4.41.1
    !pip install keras==2.3.1
    !pip install tensorflow==2.2.0
"""


def category_embedder(df, target_col):
    """
        Map categorical variables in a function approximation problem into Euclidean spaces
        Faster model training.
        Less memory overhead.
        Can give better accuracy than 1-hot encoded.
    """
    X = df.drop([target_col], axis=1)
    y = df[target_col]

    # ce.get_embedding_info identifies the categorical variables
    # of unique values and embedding size and returns a dictionary
    embedding_info = ce.get_embedding_info(X)

    X_encoded, encoders = ce.get_label_encoded_data(X)

    # splitting the data into train and Test_data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y)

    embeddings = ce.get_embeddings(X_train, y_train, categorical_embedding_info=embedding_info, is_classification=True,
                                   epochs=100, batch_size=256)

    # convert it to dataframe for easy readibility
    # dfs = ce.get_embeddings_in_dataframe(embeddings=embeddings, encoders=encoders)
    data = ce.fit_transform(X, embeddings=embeddings, encoders=encoders, drop_categorical_vars=True)
    print(data)
    return data

#
# if __name__ == '__main__':
#     df_1 = pd.DataFrame([['dog', 'xx', 'yy', 2, 1, 0, 1],
#                          ['dog', 'xx', 'yxy', 4, 6, 1, 0],
#                          ['wolf', 'uxx', 'yyx', 9, 6, 5, 0],
#                          ['rabbit', 'rxx', 'yxy', 3, 7, 4, 1],
#                          ['dog', 'rxx', 'yy', 4, 6, 1, 0],
#                          ['dog', 'axx', 'yyx', 4, 6, 1, 0],
#                          ['wolf', 'xxs', 'yy', 9, 6, 5, 0],
#                          ['rabbit', 'axx', 'yy', 3, 7, 4, 1],
#                          ['pig', 'xxx', 'yyy', 3, 9, 4, 1]
#                          ],
#                         columns=list('AGKBCDy'))
#
#     d = category_embedder(df_1, "y")
#     # print(d)
