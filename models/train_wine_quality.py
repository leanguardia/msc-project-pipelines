import argparse

import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.util.io import read_table, store_model

# def load_data(database, table):
#     df = read_table(database, table)
#     features = ['fixed acidity', 'volatile acidity', 'citric acid',
#                 'residual sugar', 'chlorides', 'free sulfur dioxide',
#                 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
#     X = df[features]
#     y = df['quality']
#     return X, y

def parse_args(args):
    if args == None:
        raise TypeError("An arguments list is required")

    model_filepath = args[0]

    if len(model_filepath) < 5 or model_filepath[-4:] != '.pkl':
        raise ValueError("Invalid model filepath")
    
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

        


    # raise TypeError(ar)
    # return {'database': 'lake/warehouse.db'}

# if __name__ == "__main__":

    # args = vars(parser.parse_args())
 
    # print('≫ Loading data')
    # database = args['database']
    # db_table = args['table']
    # X, y = load_data(database, db_table)

    # print('≫ Feature Engineering')

    # Remove Outliers
    # Polynomial Data
    # Transform to Log Scale

    # print('≫ Training Model')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
    #                                                     random_state=25)
    # model = LinearRegression()
    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    
    # # util.evaluators.eval_regression()

    # model_filepath = args['model']
    # print(f'≫ Storing Model "{model_filepath}"')
    # store_model(model, model_filepath)