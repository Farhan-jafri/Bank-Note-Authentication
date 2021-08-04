# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:06:53 2021

@author: user
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger


app = Flask(__name__)
Swagger(app)

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/') #Route Page
def welcome():
    return 'Welcome All'

@app.route('/predict',methods = ['Get'])
def predict_note_authentication():
    
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
        
      - name: variance 
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number 
        required: true
      - name: entropy
        in: query
        type: number
        required: true       
    responses:
        200:
            description: The output values
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The Predicted values is " + str(prediction)

@app.route('/predict_file',methods = ['POST'])
def predict_note_file():
    
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
    """
    
    data_test = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(data_test)
    return "The Predicted values for the csv is " + str(list(prediction))


if __name__ == '__main__':
    app.run()
    