import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('LocTreino_Equipe_8.csv')
#print(df.head())
X = df.drop(['lat', 'lon', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)
#print(X)
y = df.drop(['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

parameters = {'n_neighbors': list(range(1, 41))}
knn = KNeighborsRegressor()
grid = GridSearchCV(
    knn,
    parameters,
    cv = 10,#RepeatedKFold(n_splits = 10, n_repeats = 30),
    scoring = 'neg_mean_squared_error',
    return_train_score = True,
    refit = True
)

grid.fit(X_train, y_train)

best_knn = grid.best_estimator_
best_pred = best_knn.predict(X_test)
#print('y_test: ')
#print(y_test)
#print('best_pred: ')
#print(best_pred)
#print('mean_squared_error: ')
print(np.sqrt(mean_squared_error(y_test, best_pred)))