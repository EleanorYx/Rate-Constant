from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression
import xgboost as xgb
import pandas as pd


def select_k_best(X, y, select_num=50):
    selector = SelectKBest(score_func=f_classif, k=select_num)
    X_reduced = selector.fit_transform(X, y)

    # get selected feature index
    selected_indices = selector.get_support(indices=True)

    # print selected features
    selected_features_ = X.columns[selected_indices]
    print("selected features: ", selected_features_)

    X_ = pd.DataFrame(data=X_reduced, columns=selected_features_)
    return X_, selected_features_


def select_k_best_reg(X, y, select_num=50):
    selector = SelectKBest(score_func=f_regression(), k=select_num)
    X_reduced = selector.fit_transform(X, y)

    # get selected feature index
    selected_indices = selector.get_support(indices=True)

    # print selected features
    selected_features_ = X.columns[selected_indices]
    print("selected features: ", selected_features_)

    X_ = pd.DataFrame(data=X_reduced, columns=selected_features_)
    return X_, selected_features_


def select_by_model(X, y):
    # xgb supports null values
    model = xgb.XGBClassifier()

    model.fit(X.values, y)  # use .values to remove feature names

    # create selector
    selector = SelectFromModel(model, prefit=True)

    # get selected feature index
    selected_indices = selector.get_support(indices=True)

    # print selected features
    selected_features_ = X.columns[selected_indices]
    print("selected features: ", selected_features_)

    # transform data
    X_reduced = selector.transform(X.values)

    # create new DataFrame
    X_ = pd.DataFrame(data=X_reduced, columns=selected_features_)
    return X_, selected_features_
