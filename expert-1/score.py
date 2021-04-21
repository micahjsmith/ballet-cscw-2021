#!/usr/bin/env python3

import json

import pandas as pd
from sklearn.metrics import classification_report

from predict import predict
from settings import BUCKET

test_entities = f'https://{BUCKET}.s3.amazonaws.com/census/val/entities.csv.gz'
test_targets = f'https://{BUCKET}.s3.amazonaws.com/census/val/targets.csv.gz'

y_df = pd.read_csv(test_targets)
y_true = (y_df['PINCP'] > 84770).astype(int)

y_pred = predict(test_entities)

report = classification_report(y_true,
                               y_pred,
                               target_names=['High Income', 'Low Income'],
                               output_dict=True)

with open('./report.json', 'w') as f:
    json.dump(report, f)

print('done')
