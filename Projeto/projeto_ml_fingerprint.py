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
from sklearn.ensemble import ExtraTreesRegressor
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

def calcula_erro(grid, centro_celulas_grid, X_test, y_test, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon):

	temp = np.zeros(y_test.count()[1])
	temp2 = np.zeros((y_test.count()[1], 2))

	for k in range(0, y_test.count()[1]):
		for i in range(0, num_celulas_lat):
			for j in range(0, num_celulas_lon):
				erro_inv = 1 / np.sqrt(((grid.iloc[(i * num_celulas_lon) + j, 0] - y_test.iloc[k, 0]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 1] - y_test.iloc[k, 1]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 2] - y_test.iloc[k, 2]) ** 2)
					  		 +  ((grid.iloc[(i * num_celulas_lon) + j, 3] - y_test.iloc[k, 3]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 4] - y_test.iloc[k, 4]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 5] - y_test.iloc[k, 5]) ** 2)
					  		 +  ((grid.iloc[(i * num_celulas_lon) + j, 6] - y_test.iloc[k, 6]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 7] - y_test.iloc[k, 7]) ** 2) + ((grid.iloc[(i * num_celulas_lon) + j, 8] - y_test.iloc[k, 8]) ** 2))

				if(erro_inv > temp[k]):
					temp[k] = erro_inv
					temp2[k, 0] = centro_celulas_grid.iloc[(i * num_celulas_lon + j), 0]
					temp2[k, 1] = centro_celulas_grid.iloc[(i * num_celulas_lon + j), 1]
		#print(k)
	
	erros_pontos = np.zeros(y_test.count()[1])
	for i in range(0, y_test.count()[1]):
		erros_pontos[i] = (GeoUtils.distanceInKm(X_test.iloc[i, 0], X_test.iloc[i, 1], temp2[i, 0], temp2[i, 1]) * 1000)
	
	return temp2, erros_pontos

def plota_salva_mapa(pred, ERBS_loc, X, X_test, erro_dist, nome):

	fig, ax = plt.subplots()
	#ax.scatter(centro_celulas_grid_5['lat'], centro_celulas_grid_5['lon'], s = 1, color = 'blue', alpha = 1)
	ax.scatter(X['lon'], X['lat'], s = 10, color = 'gray', alpha = 1)
	ax.scatter(X_test['lon'], X_test['lat'], s = 10, color = 'blue', alpha = 1)
	ax.scatter(pred['lon'], pred['lat'], s = 10, color = 'green', alpha = 1)
	ax.scatter(ERBS_loc['lon'], ERBS_loc['lat'], s = 30, color = 'black', alpha = 1)
	ax.set(xlabel = 'lon', ylabel = 'lat', title = 'Erro: ' + str(erro_dist) + ' metros')
	#minor_ticks_lon = np.arange(lon_min_5, lon_max_5 + delta_lon_5, delta_lon_5)
	#minor_ticks_lat = np.arange(lat_min_5, lat_max_5 + delta_lat_5, delta_lat_5)
	#ax.set_yticks(minor_ticks_lon, minor=True)
	#ax.set_xticks(minor_ticks_lat, minor=True)
	#ax.yaxis.grid(which='minor')
	# ax.xaxis.grid(which='minor')
	# fig.savefig('graficos_projeto_ml_fingerprint/' + nome)

df = pd.read_csv('LocTreino_Equipe_8.csv')
ERBS = pd.read_csv('Bts.csv')
X = df.drop(['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)
y = df.drop(['lat', 'lon', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)
ERBS_loc = ERBS.drop(['grupo', 'btsId', 'cch', 'azimuth', 'RssiId', 'Eirp'], axis = 1)

erros = np.zeros(150) ## vetor de erros de todos pontos de teste em todas as rodadas
media_erros = 0  ## vetor de media de erro em cada rodada
pontos_pred = np.zeros((150, 2)) ## pontos preditos em cada rodada
index_coluna = ['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3']

centro_celulas_grid, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon, lat_max, lon_max = define_limites_grid(X, 5)
centro_celulas_grid = centro_celulas_grid.reshape((num_celulas_lat * num_celulas_lon), 2)
centro_celulas_grid = pd.DataFrame(centro_celulas_grid, columns = ['lat', 'lon'])

tempo_inicial = time.time()

## Para ETR 2000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 5)
etr = ExtraTreesRegressor(n_estimators = 2000)
etr.fit(X_train, y_train)
pred = pd.DataFrame(etr.predict(centro_celulas_grid), columns = index_coluna)  ## Grid predito nos centros das celulas

pontos_pred, erros = calcula_erro(pred, centro_celulas_grid, X_test, y_test, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon)

media_erros = np.mean(erros)

tempo_final = time.time()

print('ETR 2000 5 | Erro medio:', np.mean(erros), '| Erro Min:', np.min(erros), '| Erro Max:', np.max(erros), '| Desvio P.:', np.std(erros)) ## Erros medio, minimo, maximo e desvio padrao da rodada de menor erro medio
print('Tempo ETR 2000 5:', tempo_final - tempo_inicial)

## Plota o histograma normalizado da rodada de menor erro médio
fig, ax = plt.subplots()
ax.hist(erros, bins = 35, density = 1)
ax.set_xlabel('metros')
ax.set_ylabel('probabilidade')
ax.set_title('Histograma_ETR_2000_5')
# fig.savefig('graficos_projeto_ml_fingerprint/Histograma_ETR_2000_5.png')

## Plota o boxplot de todas as rodadas
fig2, ax2 = plt.subplots()
ax2.boxplot(erros)
ax2.set_ylabel('metros')
ax2.set_title('BoxPlot_ETR_2000_5')
# fig2.savefig('graficos_projeto_ml_fingerprint/BoxPlot_ETR_2000_5.png')

pred = pd.DataFrame(pontos_pred, columns = ['lat', 'lon'])
# pred.to_csv(path_or_buf = 'graficos_projeto_ml_fingerprint/ETR_2000_5.csv')

# plota_salva_mapa(pred, ERBS_loc, X, X_test, media_erros, 'Mapa_ETR_2000_5.png')

centro_celulas_grid, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon, lat_max, lon_max = define_limites_grid(X, 10)
centro_celulas_grid = centro_celulas_grid.reshape((num_celulas_lat * num_celulas_lon), 2)
centro_celulas_grid = pd.DataFrame(centro_celulas_grid, columns = ['lat', 'lon'])

tempo_inicial = time.time()

## Para ETR 2000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 5)
etr = ExtraTreesRegressor(n_estimators = 2000)
etr.fit(X_train, y_train)
pred = pd.DataFrame(etr.predict(centro_celulas_grid), columns = index_coluna)  ## Grid predito nos centros das celulas

pontos_pred, erros = calcula_erro(pred, centro_celulas_grid, X_test, y_test, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon)

media_erros = np.mean(erros)

tempo_final = time.time()

print('ETR 2000 10 | Erro medio:', np.mean(erros), '| Erro Min:', np.min(erros), '| Erro Max:', np.max(erros), '| Desvio P.:', np.std(erros)) ## Erros medio, minimo, maximo e desvio padrao da rodada de menor erro medio
print('Tempo ETR 2000 10:', tempo_final - tempo_inicial)

## Plota o histograma normalizado da rodada de menor erro médio
fig, ax = plt.subplots()
ax.hist(erros, bins = 35, density = 1)
ax.set_xlabel('metros')
ax.set_ylabel('probabilidade')
ax.set_title('Histograma_ETR_2000_10')
# fig.savefig('graficos_projeto_ml_fingerprint/Histograma_ETR_2000_10.png')

## Plota o boxplot de todas as rodadas
fig2, ax2 = plt.subplots()
ax2.boxplot(erros)
ax2.set_ylabel('metros')
ax2.set_title('BoxPlot_ETR_2000_10')
# fig2.savefig('graficos_projeto_ml_fingerprint/BoxPlot_ETR_2000_10.png')

pred = pd.DataFrame(pontos_pred, columns = ['lat', 'lon'])
# pred.to_csv(path_or_buf = 'graficos_projeto_ml_fingerprint/ETR_2000_10.csv')

# plota_salva_mapa(pred, ERBS_loc, X, X_test, media_erros, 'Mapa_ETR_2000_10.png')

centro_celulas_grid, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon, lat_max, lon_max = define_limites_grid(X, 20)
centro_celulas_grid = centro_celulas_grid.reshape((num_celulas_lat * num_celulas_lon), 2)
centro_celulas_grid = pd.DataFrame(centro_celulas_grid, columns = ['lat', 'lon'])

tempo_inicial = time.time()

## Para ETR 2000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 5)
etr = ExtraTreesRegressor(n_estimators = 2000)
etr.fit(X_train, y_train)
pred = pd.DataFrame(etr.predict(centro_celulas_grid), columns = index_coluna)  ## Grid predito nos centros das celulas

pontos_pred, erros = calcula_erro(pred, centro_celulas_grid, X_test, y_test, lat_min, lon_min, num_celulas_lat, num_celulas_lon, delta_lat, delta_lon)

media_erros = np.mean(erros)

tempo_final = time.time()

print('ETR 2000 20 | Erro medio:', np.mean(erros), '| Erro Min:', np.min(erros), '| Erro Max:', np.max(erros), '| Desvio P.:', np.std(erros)) ## Erros medio, minimo, maximo e desvio padrao da rodada de menor erro medio
print('Tempo ETR 2000 20:', tempo_final - tempo_inicial)

## Plota o histograma normalizado da rodada de menor erro médio
fig, ax = plt.subplots()
ax.hist(erros, bins = 35, density = 1)
ax.set_xlabel('metros')
ax.set_ylabel('probabilidade')
ax.set_title('Histograma_ETR_2000_20')
# fig.savefig('graficos_projeto_ml_fingerprint/Histograma_ETR_2000_20.png')

## Plota o boxplot de todas as rodadas
fig2, ax2 = plt.subplots()
ax2.boxplot(erros)
ax2.set_ylabel('metros')
ax2.set_title('BoxPlot_ETR_2000_20')
# fig2.savefig('graficos_projeto_ml_fingerprint/BoxPlot_ETR_2000_20.png')

pred = pd.DataFrame(pontos_pred, columns = ['lat', 'lon'])
# pred.to_csv(path_or_buf = 'graficos_projeto_ml_fingerprint/ETR_2000_20.csv')

# plota_salva_mapa(pred, ERBS_loc, X, X_test, media_erros, 'Mapa_ETR_2000_20.png')

# plt.show()