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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from PyRadioLoc.Utils.GeoUtils import GeoUtils
import time

def define_limites_grid(lat_lon, resolucao):
	
	lat_min = lat_lon['lat'].max() + 0.001
	lat_max = lat_lon['lat'].min() - 0.001
	lon_min = lat_lon['lon'].max() + 0.001
	lon_max = lat_lon['lon'].min() - 0.001
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
	return grid, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon, lat_max, lon_max

def calcula_erro(grid, centro_celulas_grid, X_test, y_test, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon, z, resolucao):

	temp = np.zeros(y_test.count()[1])
	temp2 = np.zeros((y_test.count()[1], 2))

	for k in range(0, y_test.count()[1]):
		for i in range(0, num_celulas_lat):
			for j in range(0, num_celulas_lon):
				erro_inv = 1 / np.sqrt(((grid.iloc[(i * num_celulas_lon) + j, 0] - y_test.iloc[k, 0]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 1] - y_test.iloc[k, 1]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 2] - y_test.iloc[k, 2]) ** 2)
					  		 +  ((grid.iloc[(i * num_celulas_lon) + j, 3] - y_test.iloc[k, 3]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 4] - y_test.iloc[k, 4]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 5] - y_test.iloc[k, 5]) ** 2)
					  		 +  ((grid.iloc[(i * num_celulas_lon) + j, 6] - y_test.iloc[k, 6]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 7] - y_test.iloc[k, 7]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 8] - y_test.iloc[k, 8]) ** 2))
				'''
				if(y_test.iloc[k, 0] == max(y_test.iloc[k, 0], y_test.iloc[k, 1], y_test.iloc[k, 2])):
					a = (y_test.iloc[k, 0] - grid.iloc[(i * num_celulas_lon) + j, 0]) ** 2
				elif(y_test.iloc[k, 1] == max(y_test.iloc[k, 0], y_test.iloc[k, 1], y_test.iloc[k, 2])):
					a = (y_test.iloc[k, 1] - grid.iloc[(i * num_celulas_lon) + j, 1]) ** 2
				else:
					a = (y_test.iloc[k, 2] - grid.iloc[(i * num_celulas_lon) + j, 2]) ** 2

				if(y_test.iloc[k, 3] == max(y_test.iloc[k, 3], y_test.iloc[k, 4], y_test.iloc[k, 5])):
					b = (y_test.iloc[k, 3] - grid.iloc[(i * num_celulas_lon) + j, 3]) ** 2
				elif(y_test.iloc[k, 4] == max(y_test.iloc[k, 3], y_test.iloc[k, 4], y_test.iloc[k, 5])):
					b = (y_test.iloc[k, 4] - grid.iloc[(i * num_celulas_lon) + j, 4]) ** 2
				else:
					b = (y_test.iloc[k, 5] - grid.iloc[(i * num_celulas_lon) + j, 5]) ** 2

				if(y_test.iloc[k, 6] == max(y_test.iloc[k, 6], y_test.iloc[k, 7], y_test.iloc[k, 8])):
					c = (y_test.iloc[k, 6] - grid.iloc[(i * num_celulas_lon) + j, 6]) ** 2
				elif(y_test.iloc[k, 7] == max(y_test.iloc[k, 6], y_test.iloc[k, 7], y_test.iloc[k, 8])):
					c = (y_test.iloc[k, 7] - grid.iloc[(i * num_celulas_lon) + j, 7]) ** 2
				else:
					c = (y_test.iloc[k, 8] - grid.iloc[(i * num_celulas_lon) + j, 8]) ** 2

				erro_inv = 1 / (np.sqrt(a + b + c))
				'''
				if(erro_inv > temp[k]):
					temp[k] = erro_inv
					temp2[k, 0] = centro_celulas_grid.iloc[(i * num_celulas_lon + j), 0]
					temp2[k, 1] = centro_celulas_grid.iloc[(i * num_celulas_lon + j), 1]
					print('trocou', z, resolucao, k, erro_inv)
	
	dist = 0
	for i in range(0, y_test.count()[1]):
		dist += (GeoUtils.distanceInKm(X_test.iloc[i, 0], X_test.iloc[i, 1], temp2[i, 0], temp2[i, 1]) * 1000)
	#print(dist/y_test.count()[1])
	
	return temp2, (dist/y_test.count()[1])

tempo_inicial = time.time()

df = pd.read_csv('LocTreino_Equipe_8.csv')
ERBS = pd.read_csv('Bts.csv')
X = df.drop(['lat', 'lon', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)
y = df.drop(['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)

dist_total_1 = 0

for k in range(10):

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
	rfr = RandomForestRegressor(n_estimators = 1000)
	rfr.fit(X_train, y_train)
	y_pred = rfr.predict(X_test)
	dist_total = 0

	for i in range(150):
		dist = GeoUtils.distanceInKm(y_test.iloc[i, 0], y_test.iloc[i, 1], y_pred[i, 0], y_pred[i, 1]) * 1000
		#print(dist)
		dist_total += dist

	dist_total_1 += (dist_total/150)
	print('media:', dist_total/150)

print('media fora:', (dist_total_1/10))

'''
centro_celulas_grid_5, lat_min_5, lon_min_5, num_celulas_lat_5, num_celulas_lon_5, delta_lat_5, delta_lon_5, lat_max_5, lon_max_5 = define_limites_grid(X, 5)
centro_celulas_grid_10, lat_min_10, lon_min_10, num_celulas_lat_10, num_celulas_lon_10, delta_lat_10, delta_lon_10, lat_max_10, lon_max_10 = define_limites_grid(X, 10)
centro_celulas_grid_15, lat_min_15, lon_min_15, num_celulas_lat_15, num_celulas_lon_15, delta_lat_15, delta_lon_15, lat_max_15, lon_max_15 = define_limites_grid(X, 15)
centro_celulas_grid_20, lat_min_20, lon_min_20, num_celulas_lat_20, num_celulas_lon_20, delta_lat_20, delta_lon_20, lat_max_20, lon_max_20 = define_limites_grid(X, 20)
centro_celulas_grid_25, lat_min_25, lon_min_25, num_celulas_lat_25, num_celulas_lon_25, delta_lat_25, delta_lon_25, lat_max_25, lon_max_25 = define_limites_grid(X, 25)

centro_celulas_grid_5 = centro_celulas_grid_5.reshape((num_celulas_lat_5 * num_celulas_lon_5), 2)
centro_celulas_grid_10 = centro_celulas_grid_10.reshape((num_celulas_lat_10 * num_celulas_lon_10), 2)
centro_celulas_grid_15 = centro_celulas_grid_15.reshape((num_celulas_lat_15 * num_celulas_lon_15), 2)
centro_celulas_grid_20 = centro_celulas_grid_20.reshape((num_celulas_lat_20 * num_celulas_lon_20), 2)
centro_celulas_grid_25 = centro_celulas_grid_25.reshape((num_celulas_lat_25 * num_celulas_lon_25), 2)

centro_celulas_grid_5 = pd.DataFrame(centro_celulas_grid_5, columns = ['lat', 'lon'])
centro_celulas_grid_10 = pd.DataFrame(centro_celulas_grid_10, columns = ['lat', 'lon'])
centro_celulas_grid_15 = pd.DataFrame(centro_celulas_grid_15, columns = ['lat', 'lon'])
centro_celulas_grid_20 = pd.DataFrame(centro_celulas_grid_20, columns = ['lat', 'lon'])
centro_celulas_grid_25 = pd.DataFrame(centro_celulas_grid_25, columns = ['lat', 'lon'])

index_coluna = ['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3']	
'''
##############################KNN############################################################################################
'''																															#
parameters = {'n_neighbors': list(range(1, 41))}																			#		
knn = KNeighborsRegressor()																									#
																															#
grid = GridSearchCV(																										#	
    knn,																													#
    parameters,																												#
    cv = 10,																												#
    scoring = 'neg_mean_squared_error',																						#
    return_train_score = True,																								#
    refit = True 																											#
)																															#
																															#
																															#
grid.fit(X_train, y_train)																									#
best_knn = grid.best_estimator_																								#
best_pred_knn = pd.DataFrame(best_knn.predict(centro_celulas_grid_5), columns = index_coluna)								#
'''																															#
#############################################################################################################################

##############################RFR############################################################################################
																															#
#rfr = RandomForestRegressor(n_estimators = 1000)																			#
#rfr.fit(X_train, y_train)																									#
																															#
#############################################################################################################################
'''
erro_dist_total_5 = 0
erro_dist_total_10 = 0
erro_dist_total_15 = 0
erro_dist_total_20 = 0
erro_dist_total_25 = 0
erro_dist_5_list = np.zeros(10)
erro_dist_10_list = np.zeros(10)
erro_dist_15_list = np.zeros(10)
erro_dist_20_list = np.zeros(10)
erro_dist_25_list = np.zeros(10)

for i in range(0, 10):

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
	rfr = RandomForestRegressor(n_estimators = 1000)
	rfr.fit(X_train, y_train)

	pred_rfr_5 = pd.DataFrame(rfr.predict(centro_celulas_grid_5), columns = index_coluna)
	pred_rfr_10 = pd.DataFrame(rfr.predict(centro_celulas_grid_10), columns = index_coluna)
	pred_rfr_15 = pd.DataFrame(rfr.predict(centro_celulas_grid_15), columns = index_coluna)
	pred_rfr_20 = pd.DataFrame(rfr.predict(centro_celulas_grid_20), columns = index_coluna)
	pred_rfr_25 = pd.DataFrame(rfr.predict(centro_celulas_grid_25), columns = index_coluna)

	pred_5, erro_dist_5 = calcula_erro(pred_rfr_5, centro_celulas_grid_5, X_test, y_test, lat_min_5, lon_min_5, num_celulas_lat_5, num_celulas_lon_5, delta_lat_5, delta_lon_5, i, 5)
	pred_10, erro_dist_10 = calcula_erro(pred_rfr_10, centro_celulas_grid_10, X_test, y_test, lat_min_10, lon_min_10, num_celulas_lat_10, num_celulas_lon_10, delta_lat_10, delta_lon_10, i, 10)
	pred_15, erro_dist_15 = calcula_erro(pred_rfr_15, centro_celulas_grid_15, X_test, y_test, lat_min_15, lon_min_15, num_celulas_lat_15, num_celulas_lon_15, delta_lat_15, delta_lon_15, i, 15)
	pred_20, erro_dist_20 = calcula_erro(pred_rfr_20, centro_celulas_grid_20, X_test, y_test, lat_min_20, lon_min_20, num_celulas_lat_20, num_celulas_lon_20, delta_lat_20, delta_lon_20, i, 20)
	pred_25, erro_dist_25 = calcula_erro(pred_rfr_25, centro_celulas_grid_25, X_test, y_test, lat_min_25, lon_min_25, num_celulas_lat_25, num_celulas_lon_25, delta_lat_25, delta_lon_25, i, 25)

	erro_dist_5_list[i] = erro_dist_5
	erro_dist_10_list[i] = erro_dist_10
	erro_dist_15_list[i] = erro_dist_15
	erro_dist_20_list[i] = erro_dist_20
	erro_dist_25_list[i] = erro_dist_25

	erro_dist_total_5 += erro_dist_5
	erro_dist_total_10 += erro_dist_10
	erro_dist_total_15 += erro_dist_15
	erro_dist_total_20 += erro_dist_20
	erro_dist_total_25 += erro_dist_25

print(erro_dist_5_list, erro_dist_total_5 / 10)
print(erro_dist_10_list, erro_dist_total_10 / 10)
print(erro_dist_15_list, erro_dist_total_15 / 10)
print(erro_dist_20_list, erro_dist_total_20 / 10)
print(erro_dist_25_list, erro_dist_total_25 / 10)

tempo_final = time.time()
print((tempo_final - tempo_inicial) / 60)
'''
'''
pred = pd.DataFrame(pred, columns = ['lat', 'lon'])
ERBS_loc = ERBS.drop(['grupo', 'btsId', 'cch', 'azimuth', 'RssiId', 'Eirp'], axis = 1)

fig, ax = plt.subplots()
ax.scatter(centro_celulas_grid_5['lat'], centro_celulas_grid_5['lon'], s = 1, color = 'blue', alpha = 1)
ax.scatter(pred['lat'], pred['lon'], s = 20, color = 'red', alpha = 1)
ax.scatter(X['lat'], X['lon'], s = 1, color = 'orange', alpha = 1)
ax.scatter(ERBS_loc['lat'], ERBS_loc['lon'], s = 30, color = 'black', alpha = 1)
#ax.scatter(X.iloc[652, 0], X.iloc[652, 1], s = 5, color = 'red', alpha = 1)
#ax.scatter(lat1, lon1, s = 5, color = 'green', alpha = 1)
#ax.scatter(lat2, lon2, s = 5, color = 'yellow', alpha = 1)
ax.set(xlabel = 'lat', ylabel = 'lon', title = 'Erro: ' + str(erro_dist) + ' metros')
#minor_ticks_lon = np.arange(lon_min_5, lon_max_5 + delta_lon_5, delta_lon_5)
#minor_ticks_lat = np.arange(lat_min_5, lat_max_5 + delta_lat_5, delta_lat_5)
#ax.set_yticks(minor_ticks_lon, minor=True)
#ax.set_xticks(minor_ticks_lat, minor=True)
#ax.yaxis.grid(which='minor')
#ax.xaxis.grid(which='minor')
#fig.savefig('ScatterErros_' + str(resolucao) + '_ML(KNN)_1.png')
plt.show()
'''