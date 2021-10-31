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
df = pd.read_csv('D:/Project/Adult-Census-Income-Prediction/adult.csv')#Vikram 
df = pd.read_csv('C:/Users/MUGESH/Desktop/project/Adult-Census-Income-Prediction/adult.csv')#Mugesh


for col in df.columns:
    print(col)

df = df.rename({'education-num': 'education_num', 'marital-status': 'marital_status', 'capital-gain': 'capital_gain', 'capital-loss': 'capital_loss', 'hours-per-week': 'hpw'}, axis=1)

#---- TO FIND ZERO VALUES IN DATA-------


print("total number of rows : {0}".format(len(df)))

for i in df.columns.values.tolist():
    print("number of rows has 0 in", i,": {0}".format(len(df.loc[df[i] == 0])))
    
    
print("number of rows missing age: {0}".format(len(df.loc[df['age'] == 0])))
print("number of rows missing workclass: {0}".format(len(df.loc[df['workclass'] == 0])))
print("number of rows missing fnlwgt: {0}".format(len(df.loc[df['fnlwgt'] == 0])))
print("number of rows missing education: {0}".format(len(df.loc[df['education'] == 0])))
print("number of rows missing education_num: {0}".format(len(df.loc[df['education_num'] == 0])))
print("number of rows missing marital_status: {0}".format(len(df.loc[df['marital_status'] == 0])))
print("number of rows missing occupation: {0}".format(len(df.loc[df['occupation'] == 0])))
print("number of rows missing relationship: {0}".format(len(df.loc[df['relationship'] == 0])))
print("number of rows missing race: {0}".format(len(df.loc[df['race'] == 0])))
print("number of rows missing sex: {0}".format(len(df.loc[df['sex'] == 0])))
print("number of rows missing capital_gain: {0}".format(len(df.loc[df['capital_gain'] == 0])))#27624
print("number of rows missing capital_loss: {0}".format(len(df.loc[df['capital_loss'] == 0])))#28735
print("number of rows missing country: {0}".format(len(df.loc[df['country'] == 0])))
print("number of rows missing salary: {0}".format(len(df.loc[df['salary'] == 0])))

# In capital gain and loss there are many zero values

#---------- Finding NA------------


for i in df.columns.values.tolist():
    if (df[i].dtype == 'O'):
        df[i] = df[i].str.strip()
    else:
        None



df.replace('?', np.nan, inplace=True)
df.isnull().mean()
df.dropna(inplace=True)


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
df['education'] = df['education'].replace('Preschool', 'Below_College').replace('1st-4th', 'Below_College').replace('5th-6th', 'Below_College').replace('7th-8th', 'Below_College').replace('9th', 'Below_College').replace('10th', 'Below_College').replace('11th', 'Below_College').replace('12th', 'Below_College').replace('HS-grad', 'Below_College').replace('Assoc-acdm', 'Below_College').replace('Assoc-voc', 'Below_College').replace('Some-college', 'College').replace('Bachelors', 'College')



#Preschool < 1st-4th < 5th-6th < 7th-8th < 9th < 10th < 11th < 12th < HS-grad < Prof-school < Assoc-acdm < Assoc-voc < Some-college < Bachelors < Masters < Doctorate.

df['education_num'].describe()

df['marital_status'].describe()
df['marital_status'].unique()
df['marital_status'] = df['marital_status'].replace('Never-married', 'Unmarried').replace('Married-civ-spouse', 'married').replace('Married-spouse-absent', 'married').replace('Married-AF-spouse', 'married').replace('Separated', 'Unmarried').replace('Widowed', 'Unmarried').replace('Divorced', 'Unmarried')


df['occupation'].describe()
df['occupation'].unique()
df['occupation'].value_counts()
#df['occupation'] = df['occupation'].replace('Handlers-cleaners', 'Others').replace('Transport-moving', 'Others').replace('Farming-fishing', 'Others').replace('Tech-support', 'Others').replace('Protective-serv', 'Others').replace('Armed-Forces', 'Others').replace('Priv-house-serv', 'Others')


df['relationship'].describe()
df['relationship'].unique()
df['relationship'].value_counts()

df['relationship'] = df['relationship'].replace('Not-in-family', 'Others').replace('Husband', 'Family').replace('Own-child', 'Family').replace('Wife', 'Family').replace('Unmarried', 'Others').replace('Other-relative', 'Others')

df['race'].describe()
df['race'].unique()
df['race'].value_counts()
df['race'] = df['race'].replace('Asian-Pac-Islander', 'Other').replace('Amer-Indian-Eskimo', 'Other')

df['sex'].describe()
df['sex'].unique()

df['capital_gain'].describe()

df['capital_loss'].describe()

df['hpw'].describe()

df['country'].describe()
df['country'].unique()
df['country'].value_counts()
df['country'] = df['country'].replace('Canada', 'APAC').replace('India', 'NAM').replace('China', 'APAC').replace('Vietnam', 'APAC').replace('Laos', 'APAC').replace('Germany', 'EMEA').replace('Portugal', 'EMEA').replace('Mexico', 'LATAM').replace('Jamaica', 'LATAM').replace('Puerto-Rico', 'LATAM').replace('Honduras', 'LATAM').replace('Cuba', 'NAM').replace('Haiti', 'EMEA').replace('Outlying-US(Guam-USVI-etc)', 'NAM').replace('Nicaragua', 'APAC').replace('Iran', 'EMEA').replace('Poland', 'EMEA').replace('Ecuador', 'LATAM').replace('Yugoslavia', 'APAC').replace('England', 'EMEA').replace('Columbia', 'LATAM').replace('Taiwan', 'APAC').replace('Dominican-Republic', 'LATAM').replace('El-Salvador', 'EMEA').replace('Guatemala', 'EMEA').replace('Italy', 'EMEA').replace('Peru', 'EMEA').replace('Trinadad&Tobago', 'EMEA').replace('Scotland', 'EMEA').replace('Greece', 'EMEA').replace('Hong', 'APAC').replace('Japan', 'APAC').replace('Philippines', 'APAC').replace('South', 'APAC').replace('France', 'EMEA').replace('Thailand', 'APAC').replace('Cambodia', 'EMEA').replace('Hungary', 'EMEA').replace('Ireland', 'EMEA').replace('Holand-Netherlands', 'EMEA')
#https://apcss.org/about/ap-countries/
#https://istizada.com/list-of-emea-countries/

df['salary'].describe()
df['salary'].unique()
df['salary'] = df['salary'].replace('<=50K', '0').replace('>50K', '1').astype('int64')

df.dtypes

sns.pairplot(df)


sns.histplot(df['fnlwgt'], kde=False).set(title = 'fnlwgt')
sns.kdeplot(df['fnlwgt'])
sns.lmplot(x='education_num', y='salary', data=df)

sns.histplot(df['education_num'], kde=False).set(title = 'education_num')
sns.kdeplot(df['education_num'])
sns.lmplot(x='fnlwgt', y='salary', data=df)

sns.histplot(df['capital_gain'], kde=False).set(title = 'capital_gain')
sns.kdeplot(df['capital_gain'])
sns.lmplot(x='capital_gain', y='salary', data=df)

sns.histplot(df['capital_loss'], kde=False).set(title = 'capital_loss')
sns.kdeplot(df['capital_loss'])
sns.lmplot(x='capital_loss', y='salary', data=df)

sns.histplot(df['hpw'], kde=False).set(title = 'hpw')
sns.kdeplot(df['hpw'])
sns.lmplot(x='hpw', y='salary', data=df)

sns.histplot(df['salary'], kde=False).set(title = 'salary')
sns.kdeplot(df['salary'])



sns.countplot(x='age', data=df).set(title = 'age')
sns.barplot(x='age',y='salary', hue = 'sex',data=df).set(title = 'age')

sns.countplot(x='workclass', data=df).set(title = 'workclass')
sns.barplot(x='workclass',y='salary', hue = 'sex',data=df).set(title = 'workclass')

sns.countplot(x='education', data=df).set(title = 'education')
sns.barplot(x='education',y='salary', hue = 'sex',data=df).set(title = 'education')

sns.countplot(x='marital_status', data=df).set(title = 'marital_status')
sns.barplot(x='marital_status',y='salary', hue = 'sex',data=df).set(title = 'marital_status')

sns.countplot(x='occupation', data=df).set(title = 'occupation')
sns.barplot(x='occupation',y='salary', hue = 'sex',data=df).set(title = 'occupation')

sns.countplot(x='relationship', data=df).set(title = 'relationship')
sns.barplot(x='relationship',y='salary', hue = 'sex',data=df).set(title = 'relationship')

sns.countplot(x='race', data=df).set(title = 'race')
sns.barplot(x='race',y='salary', hue = 'sex',data=df).set(title = 'race')

sns.countplot(x='sex', data=df).set(title = 'sex')

sns.countplot(x='country', data=df).set(title = 'country')
sns.barplot(x='country',y='salary', hue = 'sex',data=df).set(title = 'country')




def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)


sns.histplot(df['fnlwgt'], kde=False).set(title = 'fnlwgt')
sns.kdeplot(df['fnlwgt'])

sns.histplot(df['education_num'], kde=False).set(title = 'education_num')
sns.kdeplot(df['education_num'])

sns.histplot(df['capital_gain'], kde=False).set(title = 'capital_gain')
sns.kdeplot(df['capital_gain'])

sns.histplot(df['capital_loss'], kde=False).set(title = 'capital_loss')
sns.kdeplot(df['capital_loss'])

sns.histplot(df['hpw'], kde=False).set(title = 'hpw')
sns.kdeplot(df['hpw'])

sns.histplot(df['salary'], kde=False).set(title = 'salary')
sns.kdeplot(df['salary'])


df_new = pd.get_dummies(df)

   
df_new = norm_func(df_new)

#------To see the data is balanced
Less_then = len(df_new.loc[df_new['salary'] == 0])
Above_then = len(df_new.loc[df_new['salary'] == 1])
(Less_then,Above_then)


#sns.pairplot(df)

#get correlations of each features in dataset

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(40,40))
#plot heat map
g = sns.heatmap(df[top_corr_features].corr(), annot = True, cmap = "RdYlGn")



train, test = df.loc[:, df.columns != 'salary'], pd.DataFrame(df['salary'])








import dtale
d = dtale.show(df)
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

































