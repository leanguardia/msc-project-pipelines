import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures

from util.io import store_model

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("fires", engine)
    features  = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
                'month_jan', 'month_feb', 'month_mar', 'month_apr', 'month_may', 'month_jun',
                'month_jul', 'month_aug', 'month_sep', 'month_oct', 'month_nov', 'month_dec', 
                'day_mon', 'day_tue', 'day_wed', 'day_thu', 'day_fri', 'day_sat', 'day_sun']
    target = "area"
    X = df[features]
    y = df[target]
    return X, y


def evaluate(y_test, y_pred):
    print('(MAE) Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred))
    print('(MSE) Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('(R2) Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
    
#     a = np.concatenate((y_test.values.reshape(-1,1), y_pred.reshape(-1,1)), axis=1)
#     a = a[a[:,0].argsort()]

#     fig, ax = plt.subplots(figsize=(15,5))
#     x_ticks = range(y_test.shape[0])
#     ax.scatter(x_ticks, a[:,0], label='Actual', c='turquoise')
#     ax.scatter(x_ticks, a[:,1], label='Predictions', s=15, c='orange');
#     ax.legend()
# evaluate(y_test, y_pred)



# def save_model(model, model_filepath):
#     '''
#     Stores the model in a 
#     Parameters:
#         model: Sci-kit estimator object
#         model_filepath: the filepath for model storage. E.g. regressor.joblib
#     '''

#     dump(model, model_filepath)

if __name__ == "__main__":

    X, y = load_data('lake/warehouse.db')

    # Remove Outliers
    # Polynomial Data
    # Transform to Log Scale

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('(MAE) Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred))
    print('(MSE) Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('(R2) Coefficient of determination: %.2f' % r2_score(y_test, y_pred)) 

    print('Storing Regressor')
    store_model(model, 'models/regressor.pkl')
    


# Polynomial transformations
# X_train_polyer = PolynomialFeatures(degree=2)
# X_test_polyer = PolynomialFeatures(degree=2)
# X_train_poly = X_train_polyer.fit_transform(X_train)
# X_test_poly = X_test_polyer.fit_transform(X_test)
# print(X_train_polyer.get_feature_names(features))
# X_train_poly.shape

# # ### SVR

# from sklearn.preprocessing import StandardScaler
# xscaler = StandardScaler(); x_test_scaler = StandardScaler()
# yscaler = StandardScaler(); y_test_scaler = StandardScaler()
# X_train_sc = xscaler.fit_transform(X_train)
# y_train_sc = yscaler.fit_transform(y_train.values.reshape(-1,1))
# X_test_sc = x_test_scaler.fit_transform(X_test)
# y_test_sc = y_test_scaler.fit_transform(y_test.values.reshape(-1,1))

# from sklearn.svm import SVR
# svr = SVR(kernel='rbf')
# svr.fit(X_train_sc, y_train_sc.ravel())


# y_preds_sc = svr.predict(X_test_sc)
# y_preds = y_test_scaler.inverse_transform(y_preds_sc)
# y_preds[:10]

# evaluate(y_test, y_preds)

