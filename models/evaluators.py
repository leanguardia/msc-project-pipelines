import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_regression(y_test, y_pred):
    print('(MAE) Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred))
    print('(MSE) Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('(R2)  Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

def evaluate_classification(y_test, y_pred, labels=np.arange(2)):
    print("Accuracy: %.2f" % (accuracy_score(y_test, y_pred) * 100))
    print("Confusion Matrix")
    conf_mat = build_confusion_matrix(y_test, y_pred, labels)
    print(conf_mat)
    print(classification_report(y_test, y_pred))

def build_confusion_matrix(y_test, y_pred, labels):
    conf_mat = confusion_matrix(y_test, y_pred)
    high_level_ix = ['prediction'] * len(labels)
    high_level_cols = ['actuals'] * len(labels)
    return pd.DataFrame(conf_mat, index=[high_level_ix, labels],
                                  columns=[high_level_cols, labels])
