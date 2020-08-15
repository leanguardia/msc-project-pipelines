from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sklearn.externals import joblib

app = Flask(__name__)

model = joblib.load("models/forestfires.pkl")


@app.route('/')
def index():
        return render_template('index.html')


@app.route('/forestfires')
def forest_fires():
    return render_template('forestfires.html')

if __name__ == "__main__":
    app.run(debug=True)
