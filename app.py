# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:29:38 2022

@author: trist
"""

import numpy as np

import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__,template_folder='templates')
model = pickle.load(open('Best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [x for x in request.form.values()]
      
    features.append(str(4))
    
    final_features = np.array([features])
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Prediction : {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict(np.array(list(data.values())))

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)