import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    df = pd.DataFrame([np.array(int_features)], columns=['age', 'workclass', 'fnlwgt','education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hpw', 'country'])
    
    df['age'] = df['age'].astype('int64')
    df['age'] = pd.cut(df['age'], np.array([15, 25, 45, 65, 100]), 4, labels=["Young", "Middle-age", "Senior", "Old"])
    df['workclass'] = df['workclass'].replace('State-gov', 'Government').replace('Local-gov', 'Government').replace('Federal-gov', 'Government')
    df['workclass'] = df['workclass'].replace('Self-emp-not-inc', 'Self').replace('Self-emp-inc', 'Self')

    df['education'] = df['education'].replace('Preschool', 'Below_College').replace('1st-4th', 'Below_College').replace('5th-6th', 'Below_College').replace('7th-8th', 'Below_College').replace('9th', 'Below_College').replace('10th', 'Below_College').replace('11th', 'Below_College').replace('12th', 'Below_College').replace('HS-grad', 'Below_College').replace('Assoc-acdm', 'Below_College').replace('Assoc-voc', 'Below_College').replace('Some-college', 'College').replace('Bachelors', 'College')
    df['marital_status'] = df['marital_status'].replace('Never-married', 'Unmarried').replace('Married-civ-spouse', 'married').replace('Married-spouse-absent', 'married').replace('Married-AF-spouse', 'married').replace('Separated', 'Unmarried').replace('Widowed', 'Unmarried').replace('Divorced', 'Unmarried')
    df['relationship'] = df['relationship'].replace('Not-in-family', 'Others').replace('Husband', 'Family').replace('Own-child', 'Family').replace('Wife', 'Family').replace('Unmarried', 'Others').replace('Other-relative', 'Others')
    df['race'] = df['race'].replace('Asian-Pac-Islander', 'Other').replace('Amer-Indian-Eskimo', 'Other')
    df['country'] = df['country'].replace('Canada', 'APAC').replace('India', 'NAM').replace('China', 'APAC').replace('Vietnam', 'APAC').replace('Laos', 'APAC').replace('Germany', 'EMEA').replace('Portugal', 'EMEA').replace('Mexico', 'LATAM').replace('Jamaica', 'LATAM').replace('Puerto-Rico', 'LATAM').replace('Honduras', 'LATAM').replace('Cuba', 'NAM').replace('Haiti', 'EMEA').replace('Outlying-US(Guam-USVI-etc)', 'NAM').replace('Nicaragua', 'APAC').replace('Iran', 'EMEA').replace('Poland', 'EMEA').replace('Ecuador', 'LATAM').replace('Yugoslavia', 'APAC').replace('England', 'EMEA').replace('Columbia', 'LATAM').replace('Taiwan', 'APAC').replace('Dominican-Republic', 'LATAM').replace('El-Salvador', 'EMEA').replace('Guatemala', 'EMEA').replace('Italy', 'EMEA').replace('Peru', 'EMEA').replace('Trinadad&Tobago', 'EMEA').replace('Scotland', 'EMEA').replace('Greece', 'EMEA').replace('Hong', 'APAC').replace('Japan', 'APAC').replace('Philippines', 'APAC').replace('South', 'APAC').replace('France', 'EMEA').replace('Thailand', 'APAC').replace('Cambodia', 'EMEA').replace('Hungary', 'EMEA').replace('Ireland', 'EMEA').replace('Holand-Netherlands', 'EMEA')
    df_new = pd.get_dummies(df)
    def norm_func(i):
        x = (i-i.min())	/(i.max()-i.min())
        return(x)

    df_new = norm_func(df_new)
    
    
    
    
    
    
    
    prediction = model.predict(df_new)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    
    