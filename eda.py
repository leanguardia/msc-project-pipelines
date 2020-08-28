import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_corr_mat(data, title="Correlations"):
    correlations = data.corr()
    mask = np.zeros_like(correlations)
    mask[np.triu_indices_from(mask)] = True
    plt.subplots(figsize=(8, 6)); plt.title(title) 
    sns.heatmap(correlations, vmin=-1, vmax=1, mask=mask, annot=True, fmt = ".2f",
                square=True, center=0, cmap=sns.color_palette("RdBu_r", 50))

def evaluate_regression(y_test, y_pred):
    print('(MAE) Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred))
    print('(MSE) Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('(R2) Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    
def plot_regression(y_test, y_pred):
    a = np.concatenate((y_test.values.reshape(-1,1), y_pred.reshape(-1,1)), axis=1)
    a = a[a[:,0].argsort()]

    _fig, ax = plt.subplots(figsize=(15,5))
    x_ticks = range(y_test.shape[0])
    ax.scatter(x_ticks, a[:,0], label='Actual', c='turquoise')
    ax.scatter(x_ticks, a[:,1], label='Predictions', s=15, c='orange');
    ax.legend()


# def coeficient_importance(linear_model, columns, excluded_cols):
#     coeficients = pd.DataFrame([linear_model.coef_], columns=columns)
#     all_other_columns = [col for col in columns if col not in excluded_cols]
#     fig, ax = plt.subplots(figsize=(9,9))
#     coeficients[all_other_columns].plot.bar(ax=ax)