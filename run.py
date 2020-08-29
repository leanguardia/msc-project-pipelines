from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib

import pipelines
from pipelines.etl_forest_fires import ForestFireProcessor, ForestFirePredictor
from pipelines.etl_wine_quality import WineQualityProcessor
from app.form_parser import parse_wine_quality_params, parse_abalone_params, parse_adult_params

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
fires_reg = joblib.load("models/regrezz.pkl")
wine_reg = joblib.load("models/wine_reg.pkl")
abalone_reg = joblib.load("models/abalone_rgrs.pkl")
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
    processor = ForestFireProcessor()
    arry = processor.transform(args['X'], args['Y'], args['month'], args['day'],
        args['FFMC'], args['DMC'], args['DC'], args['ISI'],
        args['temp'], args['RH'], args['wind'], args['rain'])
    predictions = ForestFirePredictor(fires_reg).predict([arry])
    return str(predictions[0])

def _process_wine_quality_prediction(args):
    params = parse_wine_quality_params(args)
    processor = WineQualityProcessor()
    features = processor.transform(params)
    # TODO MAKE PREDICTOR GENERIC
    predictions = ForestFirePredictor(wine_reg).predict(features)
    return predictions[0]

def _parse_forest_fire_params(args):
    return {
        'X': float(args['X']), 'Y': float(args['Y']),
        'month': args['month'], 'day': args['day'],
        'FFMC': float(args['FFMC']), 'DMC': float(args['DMC']),
        'DC': float(args['DC']), 'ISI': float(args['ISI']),
        'temp': float(args['temp']), 'RH': float(args['RH']),
        'wind': float(args['wind']), 'rain': float(args['rain']),
    }

def _process_abalone_prediction(args):
    params = parse_abalone_params(args)
    # processor = Processor()
    # features = processor.transform(params)
    features = [params]
    predictions = ForestFirePredictor(abalone_reg).predict(features)
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
