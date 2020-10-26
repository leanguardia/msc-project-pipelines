from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

import pipelines
from pipelines.forest_fires_preparers import ForestFiresPreparer
from pipelines.predictors import RegressionPredictor
from pipelines.wine_quality_etl import WineQualityProcessor
from app.form_parser import parse_wine_quality_params, parse_abalone_params, parse_adult_params

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
fires_reg = joblib.load("models/forest_fi.pkl")
wine_reg = joblib.load("models/wine_reg.pkl")
abalone_reg = joblib.load("models/cls_abalone.pkl")
adult_cls = joblib.load("models/cls_adult.pkl")

@app.route('/')
def index():
        return render_template('index.html')

@app.route('/forestfires')
def forest_fires():
    prediction = ''
    if prediction_is_required(request):
        prediction = _process_forest_fire_prediction(request.args)
        print("And the prediction is!", prediction)
    return render_template('forestfires.html', prediction=prediction)

@app.route('/winequality')
def wine_quality():
    prediction = ''
    if prediction_is_required(request):
        prediction = _process_wine_quality_prediction(request.args)
        print("And the prediction is!", prediction)
    return render_template('winequality.html', prediction=prediction)

@app.route('/abalone')
def abalone():
    prediction = ''
    if prediction_is_required(request):
        prediction = _process_abalone_prediction(request.args)
        print("And the prediction is!", prediction)
    return render_template('abalone.html', prediction=prediction)

@app.route('/adult')
def adult():
    prediction = ''
    if prediction_is_required(request):
        prediction = _process_adult_prediction(request.args)
        print("And the prediction is!", prediction)
    return render_template('adult.html', prediction=prediction)

def _process_forest_fire_prediction(args):
    args = _parse_forest_fire_params(args)
    preparer = ForestFiresPreparer()
    arry = preparer.prepare(args)
    predictions = RegressionPredictor(fires_reg).predict(arry)
    return str(predictions[0])

def _process_wine_quality_prediction(args):
    params = parse_wine_quality_params(args)
    processor = WineQualityProcessor()
    features = processor.transform(params)
    predictions = RegressionPredictor(wine_reg).predict(features)
    return predictions[0]

def _parse_forest_fire_params(args):
    return [int(args['X']), int(args['Y']),
            args['month'], args['day'],
            float(args['FFMC']), float(args['DMC']),
            float(args['DC']), float(args['ISI']),
            float(args['temp']), float(args['RH']),
            float(args['wind']), float(args['rain']),
    ]

def _process_abalone_prediction(args):
    params = parse_abalone_params(args)
    # processor = Processor()
    # features = processor.transform(params)
    features = [params]
    predictions = RegressionPredictor(abalone_reg).predict(features)
    return predictions[0]

def _process_adult_prediction(args):
    params = parse_adult_params(args)
    # processor = Processor()
    # features = processor.transform(params)
    features = [params]
    predictions = adult_cls.predict(features) # TODO: Isolate in Predictor
    prediction = 'YES' if predictions[0] else 'NO'
    return prediction

def prediction_is_required(request):
    return len(request.args) > 0

if __name__ == "__main__":
    app.run(debug=True)
