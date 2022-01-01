import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

application = Flask(__name__)
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    int_features = pd.DataFrame([np.array(int_features)], columns=['age', 'workclass', 'fnlwgt','education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hpw', 'country'])
    #int_features.age = int_features.age.astype('int64')
    df_new = pd.get_dummies(int_features, drop_first=True)
    def norm_func(i):
        x = (i-i.min())	/(i.max()-i.min())
        return(x)

    df_new = norm_func(df_new)
    
    for col in model_columns:
          if col not in df_new.columns:
               df_new[col] = 0
    
    prediction = int(model.predict(df_new))
    if prediction == 1:
        output = 'The salary is equal or higher than 50 K'
    elif prediction == 0:
        output = 'The salary is less than 50 K'

    return render_template('index.html', prediction_text='Result: {}'.format(output))


if __name__ == "__main__":
    application.run(debug=True)
    
    
    