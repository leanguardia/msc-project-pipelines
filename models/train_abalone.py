import sys
import argparse

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from pipelines.transformation import remove_outliers_iqr, remove_outliers_zscore
from models.io import load_table, store_model, is_valid_model_filepath
from models.evaluators import evaluate_regression

def parse_args(args):
    if args == None: raise TypeError('An arguments list is required')

    model_filepath = args[0]
    if not is_valid_model_filepath(model_filepath):
        raise ValueError('Invalid model filepath {}'.format(model_filepath))

    parser = argparse.ArgumentParser(
        description='Train Abalone age predictive model.')

    parser.add_argument('model',
        help='Output model name (e.g. models/regression.pkl)')

    default_db = 'lake/warehouse.db'
    parser.add_argument('-d', '--database', dest='database', default=default_db,
        help=f'SQLite database file to query data from (default: {default_db})')
    
    default_table = 'abalones'
    parser.add_argument('-t', '--table', default=default_table,
        help=f'Database table to query (default: {default_table})')
    
    return vars(parser.parse_args(args))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
 
    print('≫ Loading data')
    df = load_table(args['database'], args['table'])

    print('≫ Feature Engineering')
    # Remove Outliers
    # Polynomial Data

    features = ['length', 'diameter', 'height', 'whole_weight',
                'shucked_weight', 'viscera_weight', 'shell_weight', 'M', 'F']
    X = df[features]
    y = df['rings']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0)
    print('≫ Training Models')
    models = {}


    print('Linear Regression')
    linreg = LinearRegression()
    parameters = {
        'fit_intercept': [True, False],
        'normalize':     [True, False],
    }
    model = GridSearchCV(linreg, param_grid=parameters)
    model.fit(X_train, y_train) 
    print('> Best parameters:', model.best_params_)
    y_pred = model.predict(X_test)

    r2 = evaluate_regression(y_test, y_pred)
    models[r2] = model.best_estimator_

    
    print('Random Forest')
    forest = RandomForestRegressor()
    parameters = {
        'n_estimators': [75, 100, 300],
        'max_depth':    [2, 5, None],
    }
    model = GridSearchCV(forest, param_grid=parameters)
    model.fit(X_train, y_train)
    print('> Best parameters:', model.best_params_)
    y_pred = model.predict(X_test)

    r2 = evaluate_regression(y_test, y_pred)
    models[r2] = model.best_estimator_


    best = models[max(models)]
    model_filepath = args['model']
    print(f'≫ Storing Model {type(best)} in "{model_filepath}"')
    store_model(best, model_filepath)
