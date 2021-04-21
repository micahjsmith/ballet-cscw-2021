from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder

from load_data import load_training_data


def upsample(x, y):
    ones_idx = np.where(y == 1)[0]
    zeros_idx = np.where(y == 0)[0]

    upsampled_ones_idx = np.random.choice(ones_idx, len(zeros_idx))

    x_ones = x.iloc[upsampled_ones_idx]
    x_zeros = x.iloc[zeros_idx]
    balanced_x = x_ones.append(x_zeros)

    y_ones = y.iloc[upsampled_ones_idx]
    y_zeros = y.iloc[zeros_idx]
    balanced_y = y_ones.append(y_zeros)

    somePerm = np.random.permutation(len(balanced_x))
    return balanced_x.iloc[somePerm], balanced_y.iloc[somePerm]


def preprocess(X_df, y_df, encoders=None):
    # fill na with somevalue say -100
    for dtype, column in zip(X_df.dtypes, X_df.columns):
        X_df[column] = X_df[column].fillna(-100)

    # drop id columns
    X_df = X_df.drop(['SERIALNO'], axis=1)

    # for every categorical variable --> one hot eoncoding
    if encoders is None:
        stage = 'fit'
        encoders = {}
    else:
        stage = 'transform'
    for dtype, column in zip(X_df.dtypes, X_df.columns):
        if dtype != "int" and dtype != "float":
            if len(Counter(X_df[column])) > 1:
                if stage == 'fit':
                    encoders[column] = (
                        OneHotEncoder(handle_unknown='ignore', sparse=False)
                        .fit(X_df[column].values.reshape(-1, 1))
                    )
                one_hot = encoders[column].transform(
                    X_df[column].values.reshape(-1, 1))
                one_hot = pd.DataFrame(index=X_df.index, data=one_hot)
                # Drop column B as it is now encoded
                X_df = X_df.drop(column, axis=1)
                # Join the encoded df
                X_df = X_df.join(one_hot, rsuffix=column)
            else:
                del X_df[column]

    return X_df, y_df, encoders


def _train(X_df, y_df):
    y_df['target'] = (y_df['PINCP'] > 84770).astype(int)
    X_df, y_df, encoders = preprocess(X_df, y_df)
    X, y = upsample(X_df, y_df['target'])
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, './model.joblib')
    joblib.dump(encoders, './encoders.joblib')
    return model, encoders


def train():
    X_df, y_df = load_training_data()
    return _train(X_df, y_df)
