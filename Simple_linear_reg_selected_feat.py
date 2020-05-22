'''In this program we take only those feature that actually affects student's grade like studytime,
previous grades etc. In the simple_linear_reg_all_in.py all features are used and hence the irrelevant
 data affects the efficiency of the model. So in this program only features that matters are used.'''

#Importing libraries
import pandas as pd
import numpy as np
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error

#Importing datasets
dataset = pd.read_csv('student-mat.csv', delimiter=';')
dataset = dataset[["G1", "G2", "studytime", "failures", "absences", "G3"]]
predict = "G3"
X = np.array(dataset.drop([predict], 1))
y = np.array(dataset[predict])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = np.array(regressor.predict(X_test))

print("Simple Linear Regression using some of the features")
#Accuracy score
acc= regressor.score(X_test,y_test)*100
print("Efficiency of the model:",acc)


