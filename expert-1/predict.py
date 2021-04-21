#!/usr/bin/env python3

import argparse
from typing import List, Union

import pandas as pd
import numpy as np

from model import train, add_intercept


def load_new_survey_responses(input_path: str) -> pd.DataFrame:
    return pd.read_csv(input_path)


def predict(input_or_path: Union[str, pd.DataFrame]) -> List[int]:
    """Make predictions for new survey responses"""
    if isinstance(input_or_path, str):
        X_df = load_new_survey_responses(input_or_path)
    else:
        X_df = input_or_path.copy()

    X_df = add_intercept(X_df)
    selected, model = train()
    X_matrix = np.array(X_df.loc(axis=1)[selected])
    Y_pred = np.dot(X_matrix, model) > 0.5
    return list(Y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict personal income.')
    parser.add_argument('input_path', metavar='input-path', type=str, help='path to CSV file containing new survey responses')
    args = parser.parse_args()
    preds = predict(args.input_path)

    with open('./predictions.txt', 'w') as f:
        f.write('\n'.join(map(str, preds)))

    print('done')
