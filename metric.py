import math
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, KFold
import sklearn.metrics as sm


def metric_model(model, train_x, train_y, test_x, test_y):
    # r2
    cv = KFold(n_splits=5, shuffle=True, random_state=100)
    scores = cross_validate(model, train_x, train_y, cv=5, scoring='accuracy')
    train_score = model.score(train_x, train_y)
    test_score = model.score(test_x, test_y)
    cv_score = scores['test_score'].mean()
    cv_std = scores['test_score'].std()
    print(f'Score:\nTrain:{train_score:.4f}, Test:{test_score:.4f}, CV:{cv_score:.4f} (+/-{cv_std})')

    # report
    pred_train_y = model.predict(train_x)
    pred_test_y = model.predict(test_x)

    print('--------------- Train -----------------')
    print(classification_report(train_y, pred_train_y, target_names=['level-1', 'level-2', 'level-3'], digits=2))
    print('--------------- Test -----------------')
    print(classification_report(test_y, pred_test_y, target_names=['level-1', 'level-2', 'level-3'], digits=2))


def metric_reg_model(model, train_x, train_y, test_x, test_y):
    # r2
    cv = KFold(n_splits=5, shuffle=True, random_state=100)
    scores = cross_validate(model, train_x, train_y, cv=cv, scoring=['r2', 'neg_root_mean_squared_error'])
    train_score = model.score(train_x, train_y)
    test_score = model.score(test_x, test_y)
    cv_score = scores['test_r2'].mean()
    cv_std = scores['test_r2'].std()
    print(f'Score: Train:{train_score:.4f}, Test:{test_score:.4f}, CV:{cv_score:.4f} (+/-{cv_std})')

    pred_train_y = model.predict(train_x)
    pred_test_y = model.predict(test_x)
    train_rmse = math.sqrt(sm.mean_squared_error(train_y, pred_train_y))
    test_rmse = math.sqrt(sm.mean_squared_error(test_y, pred_test_y))
    cv_rmse = -scores.get('test_neg_root_mean_squared_error').mean()
    cv_rmse_std = scores.get('test_neg_root_mean_squared_error').std()
    print(f'RMSE: Train:{train_rmse:.4f}, Test:{test_rmse:.4f}, CV:{cv_rmse:.4f} (+/- {cv_rmse_std:.4f})')
