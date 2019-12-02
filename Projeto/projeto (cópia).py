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
from PyRadioLoc.Utils.GeoUtils import GeoUtils

def define_limites_grid(lat_lon, resolucao):
	
	lat_min = lat_lon['lat'].max()
	lat_max = lat_lon['lat'].min()
	lon_min = lat_lon['lon'].max()
	lon_max = lat_lon['lon'].min()
	#print(lat_min, lat_max, lon_min, lon_max)

	## encontra a distancia de cobertura lateral
	dist_lat = GeoUtils.distanceInKm(lat_min, lon_min, lat_max, lon_min)
	dist_lon = GeoUtils.distanceInKm(lat_min, lon_min, lat_min, lon_max)
	#print(dist_lat, dist_lon)

	## encontra a distancia de cobertura vertical
	num_celulas_lat = int(np.ceil((dist_lat * 1000) / resolucao))
	num_celulas_lon = int(np.ceil((dist_lon * 1000) / resolucao))
	print(num_celulas_lat, num_celulas_lon)

	## encontra o delta de cada celula em termos de latitude e longitude
	delta_lat = (lat_max - lat_min) / num_celulas_lat
	delta_lon = (lon_max - lon_min) / num_celulas_lon
	#print(delta_lat, delta_lon)
	grid = np.zeros((num_celulas_lat, num_celulas_lon, 2))
	
	for i in range(0, num_celulas_lat):
		for j in range(0, num_celulas_lon):
			lat = lat_min + i * delta_lat
			lon = lon_min + j * delta_lon
			grid[i, j, 0] = lat + delta_lat / 2
			grid[i, j, 1] = lon + delta_lon / 2
	
	#print(grid[num_celulas_lat - 1, num_celulas_lon - 1, 0] + delta_lat/2, grid[num_celulas_lat - 1, num_celulas_lon - 1, 1] + delta_lon/2, lat_max, lon_max)
	return grid, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon

def calcula_erro(grid, centro_celulas_grid, X_test, y_test, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon):

	temp = np.zeros(y_test.count()[1])
	temp2 = np.zeros((y_test.count()[1], 2))

	for k in range(0, y_test.count()[1]):
		for i in range(0, num_celulas_lat):
			for j in range(0, num_celulas_lon):
				c = 1 / (abs(grid.iloc[i * 9 + j, 0] - y_test.iloc[k, 0]) + abs(grid.iloc[i * 9 + j, 1] - y_test.iloc[k, 1]) + abs(grid.iloc[i * 9 + j, 2] - y_test.iloc[k, 2])
					  +  abs(grid.iloc[i * 9 + j, 3] - y_test.iloc[k, 3]) + abs(grid.iloc[i * 9 + j, 4] - y_test.iloc[k, 4]) + abs(grid.iloc[i * 9 + j, 5] - y_test.iloc[k, 5])
					  +  abs(grid.iloc[i * 9 + j, 6] - y_test.iloc[k, 6]) + abs(grid.iloc[i * 9 + j, 7] - y_test.iloc[k, 7]) + abs(grid.iloc[i * 9 + j, 8] - y_test.iloc[k, 8]))
				if(c > temp[k]):
					temp[k] = c
					temp2[k, 0] = centro_celulas_grid.iloc[i * 9 + j, 0]
					temp2[k, 1] = centro_celulas_grid.iloc[i * 9 + j, 1]
					print('trocou', k, c)
	
	dist = 0
	for i in range(0, y_test.count()[1]):
		dist += (GeoUtils.distanceInKm(X_test.iloc[i, 0], X_test.iloc[i, 1], temp2[i, 0], temp2[i, 1]) * 1000)
	print(dist/y_test.count()[1])
	

df = pd.read_csv('LocTreino_Equipe_8.csv')
X = df.drop(['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)
y = df.drop(['lat', 'lon', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)

centro_celulas_grid_5, lat_min_5, lon_min_5, num_celulas_lat_5, num_celulas_lon_5, delta_lat_5, delta_lon_5 = define_limites_grid(X, 200)
grid_5 = np.zeros((num_celulas_lat_5, num_celulas_lon_5, 9))
centro_celulas_grid_5 = centro_celulas_grid_5.reshape((num_celulas_lat_5 * num_celulas_lon_5), 2)
centro_celulas_grid_5 = pd.DataFrame(centro_celulas_grid_5, columns = ['lat', 'lon'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
parameters = {'n_neighbors': list(range(1, 41))}
knn = KNeighborsRegressor()

grid = GridSearchCV(
    knn,
    parameters,
    cv = 10,
    scoring = 'neg_mean_squared_error',
    return_train_score = True,
    refit = True
)

index_coluna = ['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3']

grid.fit(X_train, y_train)
best_knn = grid.best_estimator_
best_pred = pd.DataFrame(best_knn.predict(centro_celulas_grid_5), columns = index_coluna)

#for i in range (0, num_celulas_lat_5):
#	for j in range (0, num_celulas_lon_5):
#		grid_5[i, j] = [best_pred.iloc[i * 9 + j, 0], best_pred.iloc[i * 9 + j, 1], best_pred.iloc[i * 9 + j, 2],
#						best_pred.iloc[i * 9 + j, 3], best_pred.iloc[i * 9 + j, 4], best_pred.iloc[i * 9 + j, 5],
#						best_pred.iloc[i * 9 + j, 6], best_pred.iloc[i * 9 + j, 7], best_pred.iloc[i * 9 + j, 8]]
#print(best_pred)
#print(y_test)
calcula_erro(best_pred, centro_celulas_grid_5, X_test, y_test, lat_min_5, lon_min_5, num_celulas_lat_5, num_celulas_lon_5, delta_lat_5, delta_lon_5)