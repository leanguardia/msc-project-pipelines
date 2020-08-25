import argparse

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures

from models.io import store_model

def load_data(database, table):
    engine = create_engine(f'sqlite:///{database}')
    df = pd.read_sql_table(table, engine)
    features  = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
                # 'month_jan', 'month_feb', 'month_mar', 'month_apr', 'month_may', 'month_jun',
                # 'month_jul', 'month_aug', 'month_sep', 'month_oct', 'month_nov', 'month_dec', 
                # 'day_mon', 'day_tue', 'day_wed', 'day_thu', 'day_fri', 'day_sat', 'day_sun'
                ]
    target = 'area'
    X = df[features]
    y = df[target]
    return X, y


def evaluate(y_test, y_pred):
    print('(MAE) Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred))
    print('(MSE) Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('(R2) Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training of Forest Fires predictive model.')
        
    parser.add_argument('model', type=str,# dest='model_filepath',
        help='Output model name (e.g. models/regression.pkl)')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database',
        type=str, dest='database', default=default_db,
        help=f'SQLite database file to query data from (default: {default_db})')

    default_table = 'fires'
    parser.add_argument('-t', '--table', type=str, default=default_table,
        help=f'Database table to query (default: {default_table})')

    args = vars(parser.parse_args())
 
    print('≫ Loading data')
    database = args['database']
    db_table = args['table']
    X, y = load_data(database, db_table)

    print('≫ Feature Engineering')

    # Remove Outliers
    # Polynomial Data
    # Transform to Log Scale

    print('≫ Training Model')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=25)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('≫ Evaluation Summary')
    print('(MAE) Mean absolute error: %.2f' % mean_absolute_error(y_test, y_pred))
    print('(MSE) Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('(R2) Coefficient of determination: %.2f' % r2_score(y_test, y_pred)) 

    model_filepath = args['model']
    print(f'≫ Storing Model "{model_filepath}"')
    store_model(model, model_filepath)

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

