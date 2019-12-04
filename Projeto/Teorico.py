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
from PyRadioLoc.Pathloss.Models import FreeSpaceModel
from PyRadioLoc.Pathloss.Models import FlatEarthModel
from PyRadioLoc.Pathloss.Models import LeeModel
from PyRadioLoc.Pathloss.Models import EricssonModel
from PyRadioLoc.Pathloss.Models import Cost231Model
from PyRadioLoc.Pathloss.Models import Cost231HataModel
from PyRadioLoc.Pathloss.Models import OkumuraHataModel
from PyRadioLoc.Pathloss.Models import Ecc33Model
from PyRadioLoc.Pathloss.Models import SuiModel

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

df = pd.read_csv('LocTreino_Equipe_8.csv')
X = df.drop(['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)
y = df.drop(['lat', 'lon', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)
ERBS = pd.read_csv('Bts.csv')
ERBS_loc = ERBS.drop(['grupo', 'btsId', 'cch', 'azimuth', 'RssiId', 'Eirp'], axis = 1)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
centro_celulas_grid_5, lat_min_5, lon_min_5, num_celulas_lat_5, num_celulas_lon_5, delta_lat_5, delta_lon_5, lat_max_5, lon_max_5 = define_limites_grid(X, 100)
grid_5 = np.zeros((num_celulas_lat_5, num_celulas_lon_5, 9))

modelo_teorico = Cost231HataModel(1800)
Eirp = 17.457772179#ERBS.iloc[0, 7]

for i in range(0, num_celulas_lat_5):
	for j in range(0, num_celulas_lon_5):
		d = GeoUtils.distanceInKm(centro_celulas_grid_5[i, j, 0], centro_celulas_grid_5[i, j, 1], ERBS_loc.iloc[0, 0], ERBS_loc.iloc[0, 1])
		pathloss = modelo_teorico.pathloss(d)
		#print(i, centro_celulas_grid_5[i, j, 0], j, centro_celulas_grid_5[i, j, 1], d)
		grid_5[i, j, 0] = Eirp - pathloss + 3
		grid_5[i, j, 1] = Eirp - pathloss + 3
		grid_5[i, j, 2] = Eirp - pathloss + 3
		d = GeoUtils.distanceInKm(centro_celulas_grid_5[i, j, 0], centro_celulas_grid_5[i, j, 1], ERBS_loc.iloc[3, 0], ERBS_loc.iloc[3, 1])
		pathloss = modelo_teorico.pathloss(d)
		#print(i, centro_celulas_grid_5[i, j, 0], j, centro_celulas_grid_5[i, j, 1], d)
		grid_5[i, j, 3] = Eirp - pathloss + 3
		grid_5[i, j, 4] = Eirp - pathloss + 3
		grid_5[i, j, 5] = Eirp - pathloss + 3
		d = GeoUtils.distanceInKm(centro_celulas_grid_5[i, j, 0], centro_celulas_grid_5[i, j, 1], ERBS_loc.iloc[6, 0], ERBS_loc.iloc[6, 1])
		pathloss = modelo_teorico.pathloss(d)
		#print(i, centro_celulas_grid_5[i, j, 0], j, centro_celulas_grid_5[i, j, 1], d)
		grid_5[i, j, 6] = Eirp - pathloss + 3
		grid_5[i, j, 7] = Eirp - pathloss + 3
		grid_5[i, j, 8] = Eirp - pathloss + 3

similaridade = 0
celula = np.zeros(3)

for i in range(0, num_celulas_lat_5):
	for j in range(0, num_celulas_lon_5):
		similaridade = 1 / GeoUtils.distanceInKm(X.iloc[652, 0], X.iloc[652, 1], centro_celulas_grid_5[i, j, 0], centro_celulas_grid_5[i, j, 1])
		#print(i, j, similaridade)
		if(similaridade >= celula[0]):
			celula[0] = similaridade
			celula[1] = i
			celula[2] = j

pred = np.zeros(3)

for i in range(0, num_celulas_lat_5):
	for j in range(0, num_celulas_lon_5):
		rssi_1 = max(y.iloc[652, 0], y.iloc[652, 1], y.iloc[652, 2])
		rssi_2 = max(y.iloc[652, 3], y.iloc[652, 4], y.iloc[652, 5])
		rssi_3 = max(y.iloc[652, 6], y.iloc[652, 7], y.iloc[652, 8])
		inv_erro = 1 / np.sqrt(((grid_5[i, j, 0] - rssi_1) ** 2) + ((grid_5[i, j, 3] - rssi_2) ** 2) + ((grid_5[i, j, 6] - rssi_3) ** 2))
		#inv_erro = 1 / np.sqrt(((grid_5[i, j, 0] - y.iloc[0, 0]) ** 2) + ((grid_5[i, j, 1] - y.iloc[0, 1]) ** 2) + ((grid_5[i, j, 2] - y.iloc[0, 2]) ** 2) + ((grid_5[i, j, 3] - y.iloc[0, 3]) ** 2) + ((grid_5[i, j, 4] - y.iloc[0, 4]) ** 2) + ((grid_5[i, j, 5] - y.iloc[0, 5]) ** 2) + ((grid_5[i, j, 6] - y.iloc[0, 6]) ** 2) + ((grid_5[i, j, 7] - y.iloc[0, 7]) ** 2) + ((grid_5[i, j, 8] - y.iloc[0, 8]) ** 2))
		#print(inv_erro)

		if(inv_erro >= pred[0]):
			pred[0] = inv_erro
			pred[1] = i
			pred[2] = j

print(rssi_1, rssi_2, rssi_3)
print(y.iloc[652, 0], y.iloc[652, 1], y.iloc[652, 2], y.iloc[652, 3], y.iloc[652, 4], y.iloc[652, 5], y.iloc[652, 6], y.iloc[652, 7], y.iloc[652, 8])
print(grid_5[int(celula[1]), int(celula[2]), 0], grid_5[int(celula[1]), int(celula[2]), 3], grid_5[int(celula[1]), int(celula[2]), 6])
print(grid_5[int(pred[1]), int(pred[2]), 0], grid_5[int(pred[1]), int(pred[2]), 3], grid_5[int(pred[1]), int(pred[2]), 6])
print(GeoUtils.distanceInKm(centro_celulas_grid_5[int(celula[1]), int(celula[2]), 0], centro_celulas_grid_5[int(celula[1]), int(celula[2]), 1], centro_celulas_grid_5[int(pred[1]), int(pred[2]), 0], centro_celulas_grid_5[int(pred[1]), int(pred[2]), 1]))

'''
pred = np.zeros((y.count()[1], 3))

for k in range(0, y.count()[1]):
	for i in range(0, num_celulas_lat_5):
		for j in range(0, num_celulas_lon_5):
			
			inv_erro = 1 / np.sqrt(((grid_5[i, j, 0] - y.iloc[k, 0]) ** 2) + ((grid_5[i, j, 1] - y.iloc[k, 1]) ** 2) + ((grid_5[i, j, 2] - y.iloc[k, 2]) ** 2) + ((grid_5[i, j, 3] - y.iloc[k, 3]) ** 2) + ((grid_5[i, j, 4] - y.iloc[k, 4]) ** 2) + ((grid_5[i, j, 5] - y.iloc[k, 5]) ** 2) + ((grid_5[i, j, 6] - y.iloc[k, 6]) ** 2) + ((grid_5[i, j, 7] - y.iloc[k, 7]) ** 2) + ((grid_5[i, j, 8] - y.iloc[k, 8]) ** 2))
			#print(k, inv_erro)

			if(inv_erro >= pred[k, 0]):
				pred[k, 0] = inv_erro
				pred[k, 1] = centro_celulas_grid_5[i, j, 0]
				pred[k, 2] = centro_celulas_grid_5[i, j, 1]

pred = pd.DataFrame(pred, columns = ['k', 'lat', 'lon'])
pred = pred.drop(['k'], axis = 1)
soma = 0

for i in range(0, y.count()[1]):
	#print(X.iloc[i, 0], X.iloc[i, 1], pred.iloc[i, 0], pred.iloc[i, 1])
	soma += GeoUtils.distanceInKm(X.iloc[i, 0], X.iloc[i, 1], pred.iloc[i, 0], pred.iloc[i, 1]) * 1000

print(soma/y.count()[1], pred.count()[1])
'''
lat1, lon1 = centro_celulas_grid_5[int(celula[1]), int(celula[2]), 0], centro_celulas_grid_5[int(celula[1]), int(celula[2]), 1]
lat2, lon2 = centro_celulas_grid_5[int(pred[1]), int(pred[2]), 0], centro_celulas_grid_5[int(pred[1]), int(pred[2]), 1]

centro_celulas_grid_5 = centro_celulas_grid_5.reshape((num_celulas_lat_5 * num_celulas_lon_5), 2)
centro_celulas_grid_5 = pd.DataFrame(centro_celulas_grid_5, columns = ['lat', 'lon'])

fig, ax = plt.subplots()
#ax.scatter(centro_celulas_grid_5['lat'], centro_celulas_grid_5['lon'], s = 1, alpha = 1)
#ax.scatter(pred['lat'], pred['lon'], s = 1, alpha = 1)
ax.scatter(X['lat'], X['lon'], s = 1, color = 'orange', alpha = 1)
ax.scatter(ERBS_loc['lat'], ERBS_loc['lon'], s = 1, color = 'black', alpha = 1)
ax.scatter(X.iloc[652, 0], X.iloc[652, 1], s = 5, color = 'red', alpha = 1)
ax.scatter(lat1, lon1, s = 5, color = 'green', alpha = 1)
ax.scatter(lat2, lon2, s = 5, color = 'yellow', alpha = 1)
ax.set(xlabel = 'lat', ylabel = 'lon')
minor_ticks_lon = np.arange(lon_min_5, lon_max_5 + delta_lon_5, delta_lon_5)
minor_ticks_lat = np.arange(lat_min_5, lat_max_5 + delta_lat_5, delta_lat_5)
ax.set_yticks(minor_ticks_lon, minor=True)
ax.set_xticks(minor_ticks_lat, minor=True)
ax.yaxis.grid(which='minor')
ax.xaxis.grid(which='minor')
#fig.savefig('ScatterErros_20.png')
plt.show()