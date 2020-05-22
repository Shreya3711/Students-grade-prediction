
''' Our dataset contains various fields with the information of students regarding their parents,background, 
failures, success, education, previous grades etc. This program basically takes all the fields and apply 
simple linear regression to predict decision boundry. Here X is the input feature and Y output feature 
whixh is divided into train set and test set. We train and produce a decision boundry with the help of train
set and use test set to judge our model. y_pred will have the predicted values of grade with respect to 
X_test'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = np.array(regressor.predict(X_test).reshape(132,1), dtype='int64')

print("Simple Linear Regression using all the features")

#Accuracy score
acc= regressor.score(X_test,y_test)*100
print("Efficiency of the model:",acc)




