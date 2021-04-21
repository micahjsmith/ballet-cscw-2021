import pandas as pd

from settings import BUCKET


def load_training_data():
    root = f'https://{BUCKET}.s3.amazonaws.com/census'
    X_df = pd.read_csv(root + '/train/entities.csv.gz')
    y_df = pd.read_csv(root + '/train/targets.csv.gz')
    return X_df, y_df


def load_test_data():
    root = f'https://{BUCKET}.s3.amazonaws.com/census'
    X_df = pd.read_csv(root + '/val/entities.csv.gz')
    y_df = pd.read_csv(root + '/val/targets.csv.gz')
    return X_df, y_df
