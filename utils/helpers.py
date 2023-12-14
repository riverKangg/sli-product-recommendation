import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score


def read_product_label():
    # read_config.py
    with open('utils/target_label_dictionary.json', 'rb') as f:
        label_dictionary = json.load(f)
    return label_dictionary


# ■■■■■■■■■ CLF : make input ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def make_base_input(df_for_clf):
    assert df_for_clf.duplicated().sum() == 0
    X_base = df_for_clf.drop(columns=['ID', 'target', 'target_category'])
    y_base = df_for_clf.target_category - 3
    assert min(y_base) == 0
    assert max(y_base) == len(set(y_base)) - 1
    return X_base, y_base


def make_chain_input(df_for_clf, dev=True):
    df_X = df_for_clf.drop(columns=['target', 'target_category']).drop_duplicates().set_index('ID')

    df_multioutput = df_for_clf[['ID', 'target_category', 'target']].pivot(index='ID',
                                                                           columns='target_category').fillna(0)
    df_multioutput.columns = [f'{col[1]}' for col in df_multioutput.columns]

    df_for_clf_chain = df_X.join(df_multioutput)
    assert df_X.shape[0] == df_multioutput.shape[0] == df_for_clf_chain.shape[0]

    X_chain = df_for_clf_chain[df_X.columns]
    y_chain = df_for_clf_chain[df_multioutput.columns]

    return X_chain, y_chain


# ■■■■■■■■■ CLF : show results ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def get_clf_pred(y_test, y_proba, output='accuracy'):
    assert output in ['accuracy', 'binary', 'category']
    assert y_proba.shape == y_test.shape
    # multiouput accuracy
    y_pred_binary = np.zeros_like(y_proba)
    y_pred_binary[np.arange(len(y_proba)), np.argmax(y_proba, axis=1)] = 1

    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f'\nAccuracy only CLF: {accuracy:.4f}')
    if output == 'accuracy':
        return accuracy
    if output == 'binary':
        return y_pred_binary

    # get category
    clf_categories = y_test.columns
    y_pred_category = [int(clf_categories[np.where(row == 1)[0]][0]) for row in y_pred_binary]
    assert y_proba.shape[0] == len(y_pred_category)

    if output == 'category':
        return y_pred_category


def get_clf_chain_result(y_test, y_proba):
    y_pred_binary = get_clf_pred(y_test, y_proba, output='binary')
    y_true = (y_test == 1) & (y_pred_binary == 1)
    df_clf_result = pd.DataFrame()
    df_clf_result['true_count'] = y_test.sum().astype(int)
    df_clf_result['pred_count'] = y_pred_binary.sum(axis=0).astype(int)
    df_clf_result['pred_true'] = y_true.sum(axis=0).astype(int)
    df_clf_result['accuracy_pred'] = round(df_clf_result['pred_true'] / df_clf_result['pred_count'] * 100, 1)
    df_clf_result['accuracy_true'] = round(df_clf_result['pred_true'] / df_clf_result['true_count'] * 100, 1)
    return df_clf_result


def get_clf_base_result(y_test, y_pred_class):
    df_clf_result = pd.DataFrame()

    df_clf_result['pred'] = list(y_pred_class) + 3
    df_clf_result['true'] = list(y_test) + 3
    df_clf_result['accuracy'] = df_clf_result['pred'] == df_clf_result['true']
    accuracy = round(sum(df_clf_result['accuracy']) / len(df_clf_result) * 100, 1)
    print(f'Accuracy: {accuracy}')

    df_clf_true_grp = df_clf_result.groupby('true')[['accuracy']].count()
    df_clf_true_grp.columns = ['true_count']

    df_clf_result_grp = df_clf_result.groupby('pred').agg({'accuracy': ['count', 'sum']})
    df_clf_result_grp.columns = ['pred_count', 'pred_true']
    df_clf_result_grp = df_clf_true_grp.join(df_clf_result_grp)
    df_clf_result_grp['accuracy'] = round(df_clf_result_grp['pred_true'] / df_clf_result_grp['pred_count'] * 100, 1)
    df_clf_result_grp['accuracy_pred'] = round(df_clf_result_grp['pred_true'] / df_clf_result_grp['pred_count'] * 100,
                                               1)
    df_clf_result_grp['accuracy_true'] = round(df_clf_result_grp['pred_true'] / df_clf_result_grp['true_count'] * 100,
                                               1)
    df_clf_result_grp[['true_count', 'pred_count']] = df_clf_result_grp[['true_count', 'pred_count']].astype(int)

    return df_clf_result_grp