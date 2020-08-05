from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)


class Fire(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    rain        = db.Column(db.Integer, default=0)
    temperature = db.Column(db.Integer, default=0)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Fire %r>' % self.id


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        rain = request.form['rain']
        temperature = request.form['temperature']
        new_fire = Fire(rain=rain, temperature=temperature)
        try:
            db.session.add(new_fire)
            db.session.commit()
            return redirect('/')
        except:
            return 'There was a Problem!'
    else:
        fires = Fire.query.order_by(Fire.created_at).all()
        return render_template('index.html', fires=fires)


if __name__ == "__main__":
    app.run(debug=True)
