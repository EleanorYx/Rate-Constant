import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from feature_select import select_k_best_reg, select_by_model
from metric import metric_model, metric_reg_model


def generate_columns(prefix, size):
    columns = []
    for i in range(size):
        columns.append(prefix + '_{}'.format(i + 1))

    return columns


def get_fps_by_mols(mols):
    x_feature = []
    for mol in mols:
        fp = MACCSkeys.GenMACCSKeys(mol)
        x_feature.append(fp.ToList())

    cols = generate_columns('MACCS', len(fp))

    return pd.DataFrame(data=x_feature, columns=cols)


def pre_process(path):
    df = pd.read_csv(path)
    bins = [0, 1e6, 1e8, np.inf]
    df['level'] = pd.cut(df['kCO3-'], bins, labels=[0, 1, 2])
    mols = []
    for smiles in df['Isomeric_SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)
    fps = get_fps_by_mols(mols)
    df_ = pd.concat([df, fps], axis=1)

    all_maccs_cols = [col for col in df_.columns if 'MACCS' in col]
    all_feature_cols = ['pH']
    all_feature_cols.extend(all_maccs_cols)

    return df_, all_feature_cols


def fit_with_classification_model(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=100)
    clf.fit(X_train, y_train)
    metric_model(clf, X_train, y_train, X_test, y_test)

    return clf


def fit_with_regression_model(X_train, X_test, y_train, y_test):
    reg = RandomForestRegressor(random_state=100)
    reg.fit(X_train, y_train)
    metric_reg_model(reg, X_train, y_train, X_test, y_test)

    return reg


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print("load data from file:", file_path)
    else:
        print("No valid dataset file path provided.")
        sys.exit(1)

    df_, all_feature_cols = pre_process(file_path)
    # 1. Classification
    # 1.1 feature_selection
    X_, _ = select_by_model(df_[all_feature_cols], df_['level'])

    # 1.2 model fitting
    X = X_.copy()
    y = df_['level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    clf = fit_with_classification_model(X_train, X_test, y_train, y_test)

    # 2. Regression
    # 2.1 feature_selection
    X_r, _ = select_k_best_reg(df_[all_feature_cols], df_['logk'], select_num=50)

    predicted_level = clf.predict(X)
    X_r['level'] = predicted_level

    # 2.2 model fitting
    X = X_r.copy()
    y = df_['logk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    fit_with_regression_model(X_train, X_test, y_train, y_test)

