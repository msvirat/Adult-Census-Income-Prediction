# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 19:51:52 2021

@author: Sathiya vigraman M
"""

'''
#Parameter selection - LogisticRegression
log_model = LogisticRegression()
param_grid = [ {'penalty' : ['l1', 'l2', 'elasticnet', 'none'], 'C' : np.logspace(-4, 4, 20), 'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'], 'max_iter' : [100, 1000,2500, 5000] } ] 
clf = RandomizedSearchCV(log_model, param_distributions = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(x_train, y_train)
shutup.please()
#Best parameter as per our input
best_clf.best_estimator_



#LogisticRegression - for train
log_model = LogisticRegression(C=0.012742749857031334, max_iter=2500, penalty='none', solver='saga')

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


pip install dtale
import dtale
d = dtale.show(df)
d.open_browser()
'''

