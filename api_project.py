from flask import Flask, render_template,request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")

def hello():
    return render_template('home1.html')

@app.route("/predict", methods=['GET','POST'])

def predict():
    if request.method == "POST":
        try:
            NewYork = float(request.form['NewYork'])
            California = float(request.form['California'])
            Florida = float(request.form['Florida'])
            Rnd_Spend = float(request.form['Rnd_Spend'])
            Admin_Spend = float(request.form['Admin_Spend'])
            Marketing_Spend = float(request.form['Marketing_Spend'])
            pred_args = [NewYork, California, Florida, Rnd_Spend, Admin_Spend, Marketing_Spend]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)
            mul_reg = open("mutiple_regr.pkl","rb")
            ml_model = joblib.load(mul_reg)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return "please check the value"

    return render_template('predict.html',prediction = model_prediction)
if __name__ == '__main__':
    app.run(debug=True)
