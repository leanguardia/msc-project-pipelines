import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from models.io import read_table, store_model
from models.evaluators import evaluate_regression

def load_data(database, table):
    df = read_table(database, table)
    features = ['fixed acidity', 'volatile acidity', 'citric acid',
                'residual sugar', 'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    X = df[features]
    y = df['quality']
    return X, y

def parse_args(args):
    if args == None:
        raise TypeError("An arguments list is required")

    model_filepath = args[0]

    if len(model_filepath) < 5 or model_filepath[-4:] != '.pkl':
        raise ValueError(f"Invalid model filepath '{model_filepath}'.")
    
    parser = argparse.ArgumentParser(
        description='Train Wine Quality predictive model.')

    parser.add_argument('model',
        help='Output model name (e.g. models/regression.pkl)')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', dest='database', default=default_db,
        help=f'SQLite database file to query data from (default: {default_db})')

    default_table = 'wines'
    parser.add_argument('-t', '--table', default=default_table,
        help=f'Database table to query (default: {default_table})')

    return vars(parser.parse_args(args))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
 
    print('≫ Loading data')
    X, y = load_data(args['database'], args['table'])

    print('≫ Feature Engineering')

    # Remove Outliers
    # Polynomial Data

    print('≫ Training Model')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=25)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)
    
    evaluate_regression(y_test, y_pred)

    model_filepath = args['model']
    print(f'≫ Storing Model "{model_filepath}"')
    store_model(linreg, model_filepath)
