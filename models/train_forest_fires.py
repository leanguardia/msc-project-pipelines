import argparse

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures

# from pipelines.dummy_transformer import remove_outliers_iqr
from models.io import store_model
from models.evaluators import evaluate_regression

def load_data(database, table):
    engine = create_engine(f'sqlite:///{database}')
    df = pd.read_sql_table(table, engine)
    features  = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'temp', 'RH', 'wind', 
                 'ISI_log', 'rain_log',
                 'apr', 'aug', 'dec', 'feb', 'jan', 'jul',
                 'jun', 'mar', 'may', 'nov', 'oct', 'sep', 
                 'fri', 'mon', 'sat', 'sun', 'thu', 'tue', 'wed'
                 ]
    target = 'area_log'
    X = df[features]
    y = df[target]
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training of Forest Fires predictive model.')
        
    parser.add_argument('model', type=str,# dest='model_filepath',
        help='Output model name (e.g. models/regression.pkl)')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database',
        type=str, dest='database', default=default_db,
        help=f'SQLite database file to query data from (default: {default_db})')

    default_table = 'forest_fires'
    parser.add_argument('-t', '--table', type=str, default=default_table,
        help=f'Database table to query (default: {default_table})')

    args = vars(parser.parse_args())
 
    print('≫ Loading data')
    database = args['database']
    db_table = args['table']
    X, y = load_data(database, db_table)

    print('≫ Model Specific Transformations')

    # Remove Outliers
    # df_full = df.copy()
    # df = remove_outliers_iqr(df, 'FFMC')
    # Polynomial Data

    print('≫ Training Model')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=25)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Invert log transformation
    y_pred = np.expm1(y_pred)
    y_test = np.expm1(y_test)
    evaluate_regression(y_test, y_pred)

    model_filepath = args['model']
    print(f'≫ Storing Model "{model_filepath}"')
    store_model(model, model_filepath)

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

