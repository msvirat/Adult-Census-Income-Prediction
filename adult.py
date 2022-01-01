


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
df = pd.read_csv('C:/Users/MUGESH/Documents/GitHub/Adult-Census-Income-Prediction/adult.csv')#Mugesh


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


#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#from sklearn.svm import SVC



from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


from datetime import datetime


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))



## Hyperparameter optimization using RandomizedSearchCV

#Log model

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score



log_model = LogisticRegression()
param_grid = [ {'penalty' : ['l1', 'l2', 'elasticnet', 'none'], 'C' : np.logspace(-4, 4, 20), 'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'], 'max_iter' : [100, 1000,2500, 5000] } ] 


random_search_log = RandomizedSearchCV(log_model, param_distributions=param_grid, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)


# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search_log.fit(x_train, y_train)
timer(start_time) # timing ends here for "start_time" variable


random_search_log.best_estimator_
random_search_log.best_params_


log_model_1 = LogisticRegression(solver= 'lbfgs', penalty= 'l2', max_iter= 5000, C =  545.559)




log_model_1.fit(x_train, y_train)
y_train_predict = log_model_1.predict(x_train)
y_predict = log_model_1.predict(x_test)
print('Train accuracy', accuracy_score(y_train, y_train_predict))#0.847
print('Test accuracy', accuracy_score(y_test, y_predict))#0.849

log_model_1 = log_model_1.fit(X,Y)
log_model_1.score(X,Y)#0.848

 


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier()

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

random_search_RF = RandomizedSearchCV(RF_model, param_distributions=param_grid, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)


# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search_RF.fit(x_train, y_train)
timer(start_time) # timing ends here for "start_time" variable


random_search_RF.best_estimator_
random_search_RF.best_params_


RF_model_1 = RandomForestClassifier(n_estimators= 33,
 min_samples_split= 2,
 min_samples_leaf= 1,
 max_features= 'sqrt',
 max_depth= 4,
 bootstrap= True)




RF_model_1.fit(x_train, y_train)
y_train_predict = RF_model_1.predict(x_train)
y_predict = RF_model_1.predict(x_test)
print('Train accuracy', accuracy_score(y_train, y_train_predict))#0.83
print('Test accuracy', accuracy_score(y_test, y_predict))#0.83

RF_model_1 = RF_model_1.fit(X,Y)
RF_model_1.score(X,Y)#0.84

 

#SVM model

from sklearn.svm import SVC

svm_model = SVC()
param_grid={'kernel': ['linear'],
      'C':np.arange(1,10,5),
      'degree':np.arange(3,6),   
      'coef0':np.arange(0.001,3,0.5),
      'gamma': ('auto', 'scale')}

random_search_svm = RandomizedSearchCV(svm_model, param_distributions=param_grid, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)


# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search_svm.fit(x_train, y_train)
timer(start_time) # timing ends here for "start_time" variable


random_search_svm.best_estimator_
random_search_svm.best_params_


svm_model_1 = SVC(kernel='linear', gamma='auto', degree=3, coef0=1.001, C=6)

svm_model_1.fit(x_train, y_train)
y_train_predict = svm_model_1.predict(x_train)
y_predict = svm_model_1.predict(x_test)
print('Train accuracy', accuracy_score(y_train, y_train_predict))#0.847
print('Test accuracy', accuracy_score(y_test, y_predict))#0.847

svm_model_1 = svm_model_1.fit(X,Y)
svm_model_1.score(X,Y)#0.847




#XG Boost


from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(x_train, y_train)
y_train_predict = model.predict(x_train)
y_predict = model.predict(x_test)
print('Train accuracy', accuracy_score(y_train, y_train_predict))
print('Test accuracy', accuracy_score(y_test, y_predict))


## Hyper Parameter Optimization

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

}



classifier = XGBClassifier()


random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)


# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(x_train, y_train)
timer(start_time) # timing ends here for "start_time" variable


random_search.best_estimator_
random_search.best_params_





model_1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7,
              enable_categorical=False, gamma=0.3, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.25, max_delta_step=0, max_depth=4,
              min_child_weight=5, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1, predictor='auto',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)


model_1.fit(x_train, y_train)
y_train_predict = model_1.predict(x_train)
y_predict = model_1.predict(x_test)
print('Train accuracy', accuracy_score(y_train, y_train_predict))#0.873678
print('Test accuracy', accuracy_score(y_test, y_predict))#0.867395

model_1 = model_1.fit(X,Y)
model_1.score(X,Y)#0.8731184

 

import pickle

pickle.dump(model_1, open('model.pkl', 'wb'))


Dumm = pd.DataFrame([215], columns=['Name'])
Dumm_dumm = pd.get_dummies(Dumm)


def norm_func(i):
    x = (i-i.min())	/(i.max()-i.min())
    return(x)

Dumm_dumm = norm_func(Dumm)

x_train.columns
model_1.predict(dummm)
df_new.iloc[-1]

dummm = pd.DataFrame(x_train.iloc[-1:,])

df['age'].astype(int)

