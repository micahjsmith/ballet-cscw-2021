#!/usr/bin/env python3

import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from load_data import load_test_data
from predict import predict

X_df, y_df = load_test_data()
y_pred = predict(X_df)
y_true = (y_df['PINCP'] > 84770).astype(int)

# drop nans
nans = np.isnan(y_pred).ravel()
failures = np.sum(nans)
y_true_clean, y_pred_clean = y_true[~nans], y_pred[~nans]
report = classification_report(y_true_clean,
                               y_pred_clean,
                               target_names=['High Income', 'Low Income'],
                               output_dict=True)
report['failures'] = failures / len(y_true)

with open('./report.json', 'w') as f:
    json.dump(report, f)

print('done')
