import sys
import argparse

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures

from pipelines.transformation import remove_outliers_iqr
from models.io import store_model, is_valid_model_filepath
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

def parse_args(args):
    if args == None: raise TypeError('An arguments list is required')

    model_filepath = args[0]
    if not is_valid_model_filepath(model_filepath):
        raise ValueError('Invalid model filepath {}'.format(model_filepath))

    parser = argparse.ArgumentParser(
        description='Train Adult income predictive model.')

    parser.add_argument('model',
        help='Output model name (e.g. models/regressor.pkl)')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', dest='database', default=default_db,
        help=f'SQLite database file to query data from (default: {default_db})')

    default_table = 'forest_fires'
    parser.add_argument('-t', '--table', default=default_table,
        help=f'Database table to query (default: {default_table})')

    return vars(parser.parse_args(args))

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
 
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

