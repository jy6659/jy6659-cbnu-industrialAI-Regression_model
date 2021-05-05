# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:44:11 2021

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pit
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/Data/jf_data.csv') 
df.head()

data = df.values
X, y = data[:, :-1], data[:, -1]

sc_x = StandardScaler()
sc_y = StandardScaler() 
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.2, random_state=123)

linear = LinearRegression()
ridge = Ridge(alpha = 1.0, random_state=0)
lasso = Lasso(alpha=1.0, random_state=0)
enet = ElasticNet(alpha=1.0, l1_ratio=0.5)
linear.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
enet.fit(X_train, y_train)

linear_pred = linear.predict(X_train)
ridge_pred = ridge.predict(X_train)
lasso_pred = lasso.predict(X_train)
enet_pred = enet.predict(X_train)
print('Linear - RMSE for training data: ', np.sqrt(mean_squared_error(y_train, linear_pred)))
print('Ridge - RMSE for training data: ', np.sqrt(mean_squared_error(y_train, ridge_pred)))
print('Lasso - RMSE for training data: ', np.sqrt(mean_squared_error(y_train, lasso_pred)))
print('Elastic Net - RMSE for training data: ', np.sqrt(mean_squared_error(y_train, enet_pred)))

linear_pred = linear.predict(X_test)
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)
enet_pred = enet.predict(X_test)
print('\nLinear - RMSE for test data: ', np.sqrt(mean_squared_error(y_test, linear_pred)))
print('Ridge - RMSE for test data: ', np.sqrt(mean_squared_error(y_test, ridge_pred)))
print('Lasso - RMSE for test data: ', np.sqrt(mean_squared_error(y_test, lasso_pred)))
print('Elastic Net - RMSE for test data: ', np.sqrt(mean_squared_error(y_test, enet_pred)))

# print('Accuracy of Linear regression classifier on test set: {:.2f}'.format(LinearRegression.score(X_test, y_test)))
plt.scatter(y_test, linear_pred, alpha=0.4)
plt.plot(X_train, y_test, 'o')
plt.plot(X_train,linear.predict(X_train.values.reshape(-1,1)))
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Linear")
plt.show()

print('Accuracy of Linear regression classifier on test set: {:.2f}'.format(ridge.score(X_test, y_test)))
plt.scatter(y_test, ridge_pred, alpha=0.4)
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Ridge")
plt.show()

print('Accuracy of Linear regression classifier on test set: {:.2f}'.format(lasso.score(X_test, y_test)))
plt.scatter(y_test, lasso_pred, alpha=0.4)
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Lasso")
plt.show()

print('Accuracy of Linear regression classifier on test set: {:.2f}'.format(ElasticNet.score(X_test, y_test)))
plt.scatter(y_test, enet_pred, alpha=0.4)
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Elastic Net")
plt.show()