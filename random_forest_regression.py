'''In this program random forest regression is used and it is one of the best model and gives great accuracy
on seen and unseen data. It automatically decides the importance and weight of each feature. It generalizes
well over thhe unseen data.'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
from sklearn.metrics import mean_squared_error

# Importing the dataset
dataset = pd.read_csv('student-mat.csv', delimiter=';')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1 ].values

#Convert categorical data into numerical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype='int64')

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)


# Predicting the Test set results
y_pred = np.array(regressor.predict(X_test))

print("Random Forest Regression")

#Accuracy score
acc= regressor.score(X_test,y_test)*100
print("Efficiency of the model:",acc)
