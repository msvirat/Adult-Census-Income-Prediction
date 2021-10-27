# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 08:57:16 2021

@author: Naushina Farheen S
"""

import pandas as pd# deals with data frame 
import numpy as np# deals with numerical values
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import seaborn as sns
import dtale as dt
import os
'''
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
'''
df = pd.read_csv("C:/Users/Naushina Farheen S/Dropbox/My PC (DESKTOP-4RG72G3)/Documents/GitHub/Adult-Census-Income-Prediction/adult.csv")# For Loading csv data.



for col in df.columns:
    print(col)

df = df.rename({'education-num': 'education_num', 'marital-status': 'marital_status', 'capital-gain': 'capital_gain', 'capital-loss': 'capital_loss', 'hours-per-week': 'hpw'}, axis=1)


df.replace(' ?', np.nan, inplace=True)
df.isnull().mean()
df.dropna(inplace=True)


for i in df.columns.values.tolist():
    if (df[i].dtype == 'O'):
        df[i] = df[i].str.strip()
    else:
        None


for col in df.columns:
    print(col)



df['age'].describe()
df['age'] = pd.cut(df['age'], np.array([15, 25, 45, 65, 100]), 4, labels=["Young", "Middle-age", "Senior", "Old"])



df['workclass'].describe()
df['workclass'].unique()
df['workclass'] = df['workclass'].replace('State-gov', 'Government').replace('Local-gov', 'Government').replace('Federal-gov', 'Government')
df['workclass'] = df['workclass'].replace('Self-emp-not-inc', 'Self').replace('Self-emp-inc', 'Self')



df['fnlwgt'].describe()


df['education'].describe()
df['education'].unique()
df['education'].value_counts()
sns.histplot(df['education'])
plt.plot(df['education'])
df['education'] = df['education'].replace('Preschool', 'Below_College').replace('1st-4th', 'Below_College').replace('5th-6th', 'Below_College').replace('7th-8th', 'Below_College').replace('9th', 'Below_College').replace('10th', 'Below_College').replace('11th', 'Below_College').replace('12th', 'Below_College').replace('HS-grad', 'Below_College').replace('Assoc-acdm', 'Below_College').replace('Assoc-voc', 'Below_College')

#Degree of college?

#Preschool < 1st-4th < 5th-6th < 7th-8th < 9th < 10th < 11th < 12th < HS-grad < Prof-school < Assoc-acdm < Assoc-voc < Some-college < Bachelors < Masters < Doctorate.

df['education_num'].describe()

df['marital_status'].describe()
df['marital_status'].unique()
df['marital_status'] = df['marital_status'].replace('Never-married', 'Unmarried').replace('Married-civ-spouse', 'married').replace('Married-spouse-absent', 'married').replace('Married-AF-spouse', 'married').replace('Separated', 'Unmarried').replace('Widowed', 'Unmarried').replace('Divorced', 'Unmarried')


df['occupation'].describe()
df['occupation'].unique()
df['occupation'] = df['occupation'].replace('Adm-clerical', 'Government').replace('Exec-managerial', 'Government').replace('Handlers-cleaners', 'Private').replace('Prof-specialty', 'Private').replace('Other-service', 'Private').replace('Sales', 'Private').replace('Transport-moving', 'Private').replace('Farming-fishing', 'Private').replace('Machine-op-inspct', 'Government').replace('Tech-support', 'Private').replace('Craft-repair', 'Private').replace('Protective-serv', 'Government').replace('Armed-Forces', 'Government').replace('Priv-house-serv', 'Private')


df['relationship'].describe()
df['relationship'].unique()
df['relationship'] = df['relationship'].replace('Not-in-family', 'Others').replace('Husband', 'Family').replace('Own-child', 'Family').replace('Wife', 'Family').replace('Unmarried', 'Others').replace('Other-relative', 'Others')

df['race'].describe()
df['race'].unique()
df['race'] = df['race'].replace('Asian-Pac-Islander', 'Other').replace('Amer-Indian-Eskimo', 'Other')

df['sex'].describe()
df['sex'].unique()

df['capital_gain'].describe()

df['capital_loss'].describe()

df['hpw'].describe()

df['country'].describe()
df['country'].unique()
df['country'] = df['country'].replace('United-States', 'APAC').replace('Canada', 'APAC').replace('India', 'NAM').replace('China', 'APAC').replace('Vietnam', 'APAC').replace('Laos', 'APAC').replace('Germany', 'EMEA').replace('Portugal', 'EMEA').replace('Mexico', 'LATAM').replace('Jamaica', 'LATAM').replace('Puerto-Rico', 'LATAM').replace('Honduras', 'LATAM').replace('Cuba', 'NAM').replace('Haiti', 'EMEA').replace('Outlying-US(Guam-USVI-etc)', 'NAM').replace('Nicaragua', 'APAC').replace('Iran', 'EMEA').replace('Poland', 'EMEA').replace('Ecuador', 'LATAM').replace('Yugoslavia', 'APAC').replace('England', 'EMEA').replace('Columbia', 'LATAM').replace('Taiwan', 'APAC').replace('Dominican-Republic', 'LATAM').replace('El-Salvador', 'EMEA').replace('Guatemala', 'EMEA').replace('Italy', 'EMEA').replace('Peru', 'EMEA').replace('Trinadad&Tobago', 'EMEA').replace('Scotland', 'EMEA').replace('Greece', 'EMEA').replace('Hong', 'APAC').replace('Japan', 'APAC').replace('Philippines', 'APAC').replace('South', 'APAC').replace('France', 'EMEA').replace('Thailand', 'APAC').replace('Cambodia', 'EMEA').replace('Hungary', 'EMEA').replace('Ireland', 'EMEA').replace('Holand-Netherlands', 'EMEA')

df['salary'].describe()





df = pd.DataFrame(df)

df.describe()

df.var()
df.skew()

import dtale
dtale.show(df)
d.open_browser()


#### Dtale
d=dt.show(df)
d.open_browser()

df['occupation'].unique()
df['country'].unique()

df.mode()

plt.hist(df["output variable"])  #right skewed =0.647
plt.hist(XYZ["input variable"])  #right skewed =0.858
plt.boxplot(XYZ["output variable"],0,"rs",0) #Graphical representation for outliers present
plt.boxplot(XYZ["input variable"],0,"rs",0) 
#plt.bar(output = XYZ, x = np.arange(1, 110, 1))
plt.hist(XYZ) #histogram
plt.boxplot(XYZ) #boxplot

