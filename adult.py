# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 08:57:16 2021

@author: Naushina Farheen S
"""

import pandas as pd# deals with data frame 
import numpy as np# deals with numerical values
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import seaborn as sns



'''
import os

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



df_new = pd.get_dummies(df, drop_first=True)

   


def norm_func(i):
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

df_new = norm_func(df_new)

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



X, Y = df_new.loc[:, df_new.columns != 'salary'], pd.DataFrame(df_new['salary'])

#----Model Selection----
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import shutup



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


#Parameter selection - LogisticRegression
log_model = LogisticRegression()
param_grid = [ {'penalty' : ['l1', 'l2', 'elasticnet', 'none'], 'C' : np.logspace(-4, 4, 20), 'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'], 'max_iter' : [100, 1000,2500, 5000] } ] 
clf = GridSearchCV(log_model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(x_train, y_train)
shutup.please()
#Best parameter as per our input
best_clf.best_estimator_



#LogisticRegression - for train
log_model = LogisticRegression(C=0.0001, penalty='none', solver='sag')
get_score(log_model, x_train, x_test, y_train, y_test)#0.8491629371788496



#Random forest HyperParameter selection

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)

random_forest = RandomForestClassifier()
rf_Grid = GridSearchCV(estimator = random_forest, param_grid = param_grid, verbose=2, n_jobs = 4)
rf_Grid.fit(x_train, y_train)

rf_Grid.best_params_

print (f'Train Accuracy - : {rf_Grid.score(x_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(x_test,y_test):.3f}')



vc = VotingClassifier([('clf1', log_model), ('clf2', rf_Grid)], voting='soft')
cross_val_score(vc, X, Y).mean()






svm = SVC()
param_grid={'kernel': ['linear'],
      'C':np.arange(1,10,5),
      'degree':np.arange(3,6),   
      'coef0':np.arange(0.001,3,0.5),
      'gamma': ('auto', 'scale')}
svm_Grid = GridSearchCV(estimator = svm, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)

svm_Grid.fit(x_train, np.ravel(y_train,order='C'))

svm_Grid.best_params_

print (f'Train Accuracy - : {svm_Grid.score(x_train,y_train):.3f}')
print (f'Test Accuracy - : {svm_Grid.score(x_test,y_test):.3f}')



param_grid={'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
      'C':np.arange(1,42,10),
      'degree':np.arange(3,6),   
      'coef0':np.arange(0.001,3,0.5),
      'gamma': ('auto', 'scale')}



#RandomForestClassifier - for train model

random_forest = RandomForestClassifier(n_estimators=40)
get_score(random_forest, x_train, x_test, y_train, y_test)


#SVC - for train model
svm = SVC()
get_score(svm, x_train, x_test, y_train, y_test)


#K-Fold - model Selection

folds = StratifiedKFold(n_splits=5)

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(X,Y):
    x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
    scores_logistic.append(get_score(LogisticRegression(C=0.0001, penalty='none', solver='sag'), x_train, x_test, y_train, y_test)) 
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), x_train, x_test, y_train, y_test))
    scores_svm.append(get_score(SVC(gamma='auto'), x_train, x_test, y_train, y_test))


scores_logistic

scores_svm

scores_rf


#cross_val_score for X and Y
cross_val_score(LogisticRegression(C = 0.0001, penalty = 'none', solver='sag'), x_test, y_test, cv = 5)
cross_val_score(RandomForestClassifier(n_estimators=40), x_test, y_test, cv = 5)
cross_val_score(SVC(gamma='auto'), x_test, y_test, cv = 5)


#LogisticRegression - For main Data

log_model = LogisticRegression(C = 0.0001, penalty = 'none', solver='sag')
log_model = log_model.fit(X, Y)

log_model.score(X, Y)

#-----Creating pipeline--------


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


#pipeline_lr=Pipeline([('lr_classifier',LogisticRegression(random_state=0))])
#pipeline_rf=Pipeline([('rf_classifier',RandomForestClassifier(random_state=0))])


pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)),
                     ('lr_classifier',LogisticRegression(random_state=0))])


pipeline_dt=Pipeline([('scalar2',StandardScaler()),
                     ('pca2',PCA(n_components=2)),
                     ('dt_classifier',DecisionTreeClassifier())])

pipeline_randomforest=Pipeline([('scalar3',StandardScaler()),
                     ('pca3',PCA(n_components=2)),
                     ('rf_classifier',RandomForestClassifier())])

pipelines = [pipeline_lr, pipeline_dt,pipeline_randomforest]
best_accuracy=0.0
best_classifier=0
best_pipeline=""

pipe_dict = {0: 'Logistic Regression', 1: 'DecisionTree',2:'RandomForest'}

# Fit the pipelines
for pipe in pipelines:
	pipe.fit(x_train, y_train)

for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(x_test,y_test)))


for i,model in enumerate(pipelines):
    if model.score(x_test,y_test)>best_accuracy:
        best_accuracy=model.score(x_test,y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))


#-----Hyperparameter tuning---------
# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])#Installing the pipeline
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(0, 4, 10)
                 },
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2'],
                 "classifier__C": np.logspace(0, 4, 10),
                 "classifier__solver":['newton-cg','saga','sag','liblinear'] ##This solvers don't allow L1 penalty
                 },
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__min_samples_leaf":[1,2,5,10,15,100],
                 "classifier__max_leaf_nodes": [2, 5,10]}]
# create a gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(x_train,y_train)

print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(x_test,y_test))

#----lasso and Ridge regression

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,x_train,y_train,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x_train,y_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(x_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

prediction_lasso=lasso_regressor.predict(x_test)
prediction_ridge=ridge_regressor.predict(x_test)
sns.distplot(y_test-prediction_lasso.reshape(6032,1))
sns.distplot(y_test-prediction_ridge)


import dtale
d = dtale.show(df)
d.open_browser()

































