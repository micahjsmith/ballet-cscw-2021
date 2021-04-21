import random

import funcy as fy
import numpy as np

from load_data import load_training_data


def is_constant(array):
    if len(array) == 0:
        return True
    else:
        value = array[0]
        for entry in array:
            if value != entry:
                return False
        return True


def non_constant_col_labels(X_df):
    col_labels = []
    for col in X_df.columns:
        if (not is_constant(X_df[col])) and X_df[col].dtype in {np.dtype("float64"), np.dtype("int64")}:
            if not X_df[col].isnull().any():
                col_labels.append(col)
    return col_labels


def train_test_val_split(n, train_ratio = 0.6, test_ratio = 0.2):
    random.seed(0)
    train_idx = []
    test_idx = []
    val_idx = []
    for i in range(n):
        rand = random.random()
        if rand > train_ratio + test_ratio:
            val_idx.append(i)
        elif rand > train_ratio:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, test_idx, val_idx


def add_intercept(array_or_df):
    """Add intercept as last column"""
    length = array_or_df.shape[0]
    intercept = np.array([[1]*length]).T
    if isinstance(array_or_df, np.ndarray):
        return np.concatenate([array_or_df, intercept], axis = 1)
    else:
        return array_or_df.assign(_intercept=intercept)


def greedy_forward_selection_train(
    X_train,
    Y_train,
    X_test,
    Y_test,
    l2 = 0,
    overfitting_prevention = 0.8,
    min_inc = 0.005
):
    """
    In each iteration, select the feature such that using (selected + this feature) training on train
    and testing on test has the highest performance with the restriction
    (test l2 loss > overfitting_prevention * training l2 loss)
    """
    idx_selected = []
    idx_univ = set(range(X_train.shape[1]))
    XTX_train = np.dot(X_train.T, X_train)
    XTY_train = np.dot(X_train.T, Y_train)
    best_perf = float("inf")
    best_model = None
    while idx_univ:
        best_feature = None
        best_perf_temp = best_perf
        best_model_temp = None
        for feature in idx_univ:
            idx = idx_selected + [feature]
            XTX_real = XTX_train[idx, :][:, idx]
            XTY_real = XTY_train[idx]
            try:
                model = np.dot(np.linalg.inv(XTX_real), XTY_real)
            except:
                continue
            X_real = X_test[:, idx]
            Y_pred = np.dot(X_real, model)
            l2_test = np.mean((Y_pred-Y_test)**2)

            X_train_real = X_train[:, idx]
            Y_pred_train = np.dot(X_train_real, model)
            l2_train = np.mean((Y_pred_train - Y_train)**2)

            if l2_train*overfitting_prevention < l2_test and l2_test < best_perf_temp:
                best_perf_temp = l2_test
                best_feature = feature
                best_model_temp = model
        if best_perf_temp < (1-min_inc)*best_perf:

            idx_selected.append(best_feature)
            idx_univ.remove(best_feature)
            best_model = best_model_temp
            best_perf = best_perf_temp
            print(best_perf)
        else:
            break

    return idx_selected, best_model


def _test_model(Y_pred, Y_val):
    diff = sum(abs(Y_pred - Y_val))
    count = len(Y_val)
    precision = 1 - diff/count
    return precision, Y_pred


def test_target_model(idx, model, X_val, Y_val):
    Y_pred = np.dot(X_val[:, idx], model) > 0.5
    return _test_model(Y_pred, Y_val)


def test_income_model(idx, model, X_val, Y_val):
    Y_pred = np.dot(X_val[:, idx], model) > 84770
    return _test_model(Y_pred, Y_val)


def recover_columns(selected, cols):
    return [cols[i] for i in selected]


def _train(X_df, y_df):
    """Returns model which is a tuple of selected columns and corresponding parameters"""

    train, test, val = train_test_val_split(len(X_df))

    col_labels = non_constant_col_labels(X_df)
    X_train = np.array(X_df[col_labels])[train]
    X_test = np.array(X_df[col_labels])[test]
    X_val = np.array(X_df[col_labels])[val]

    X_train = add_intercept(X_train)
    X_test = add_intercept(X_test)
    X_val = add_intercept(X_val)

    #simple model train using target
    target = (y_df['PINCP'] > 84770).astype(int)
    Y_train = np.array(target[train])
    Y_test = np.array(target[test])
    Y_val = np.array(target[val])

    #simple model train using income
    income = y_df['PINCP']
    Y_train_income = np.array(income[train])
    Y_test_income = np.array(income[test])

    selected_target, model_target = greedy_forward_selection_train(
        X_train, Y_train, X_test, Y_test)
    selected_income, model_income = greedy_forward_selection_train(
        X_train, Y_train_income, X_test, Y_test_income)

    precision_target, _ = test_target_model(
        selected_target, model_target, X_val, Y_val)
    precision_income, _ = test_income_model(
        selected_income, model_income, X_val, Y_val)

    if precision_target >= precision_income:
        return recover_columns(selected_target, col_labels), model_target
    else:
        return recover_columns(selected_income, col_labels), model_income


@fy.memoize
def train():
    X_df, y_df = load_training_data()
    return _train(X_df, y_df)
