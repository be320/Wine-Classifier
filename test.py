import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib



dataset = './winequality-red.csv'
data = pd.read_csv(dataset, sep=';')

y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print (X_test_scaled.mean(axis=0))
print("-----------------------------------------")
print (X_test_scaled.std(axis=0))
print("-----------------------------------------")

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

print (pipeline.get_params())
print("-----------------------------------------")

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)
print (clf.best_params_)
print("-----------------------------------------")


print (clf.refit)
print("-----------------------------------------")

y_pred = clf.predict(X_test)
print (r2_score(y_test, y_pred))
print("-----------------------------------------")

print (mean_squared_error(y_test, y_pred))
print("-----------------------------------------")
	
joblib.dump(clf, 'rf_regressor.pkl')

#to load the model:
# clf2 = joblib.load('rf_regressor.pkl')
# clf2.predict(X_test)