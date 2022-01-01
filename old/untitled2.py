import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

app = Flask(__name__)
# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 14)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    to_predict_list = request.form.to_dict()
    
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    result = ValuePredictor(to_predict_list)       
    if int(result)== 1:
        prediction ='Income more than 50K'
    else:
        prediction ='Income less that 50K'           
        
    
    
    
    

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
    
    