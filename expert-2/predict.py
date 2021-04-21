#!/usr/bin/env python3

import argparse
import os.path
from contextlib import suppress
from typing import List, Union

import joblib
import numpy as np
import pandas as pd

from model import preprocess, train


def load_new_survey_responses(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)


def safe_predict(model, X):
    y = np.full((X.shape[0], 1), np.nan)
    for i in range(X.shape[0]):
        x = X.iloc[i:i+1]
        with suppress(Exception):
            y[i] = model.predict(x)
    return y


def predict(input_or_path: Union[str, pd.DataFrame]) -> List[int]:
    """Make predictions for new survey responses"""
    if isinstance(input_or_path, str):
        X_df = load_new_survey_responses(input_or_path)
    else:
        X_df = input_or_path.copy()

    if not os.path.exists('./model.joblib') \
            or not os.path.exists('./encoders.joblib'):
        model, encoders = train()
    else:
        model = joblib.load('./model.joblib')
        encoders = joblib.load('./encoders.joblib')

    X, _, _ = preprocess(X_df, None, encoders=encoders)
    return safe_predict(model, X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict personal income.')
    parser.add_argument(
        'input_path', metavar='input-path', type=str,
        help='path to CSV file containing new survey responses')
    args = parser.parse_args()
    preds = predict(args.input_path)

    with open('./predictions.txt', 'w') as f:
        f.write('\n'.join(map(str, preds)))

    print('done')
