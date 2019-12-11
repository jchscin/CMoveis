import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from PyRadioLoc.Utils.GeoUtils import GeoUtils
import time

df = pd.read_csv('LocTreino_Equipe_8.csv')
ERBS = pd.read_csv('Bts.csv')
ERBS_loc = ERBS.drop(['grupo', 'btsId', 'cch', 'azimuth', 'RssiId', 'Eirp'], axis = 1)
X = df.drop(['lat', 'lon', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)
y = df.drop(['rssi_1_1', 'rssi_1_2', 'rssi_1_3', 'rssi_2_1', 'rssi_2_2', 'rssi_2_3', 'rssi_3_1', 'rssi_3_2', 'rssi_3_3', 'delay_1', 'delay_2', 'delay_3', 'pontoId'], axis = 1)

erros = np.zeros(150) ## vetor de erros de todos pontos de teste em todas as rodadas
media_erros = 0  ## vetor de media de erro em cada rodada
pontos_pred = np.zeros((150, 2)) ## pontos preditos em cada rodada

tempo_inicial = time.time()

## Para RFR 1000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 5)
rfr = RandomForestRegressor(n_estimators = 1000)
rfr.fit(X_train, y_train)
pontos_pred = rfr.predict(X_test)

for i in range(150):
	erros[i] = GeoUtils.distanceInKm(y_test.iloc[i, 0], y_test.iloc[i, 1], pontos_pred[i, 0], pontos_pred[i, 1]) * 1000

media_erros = np.mean(erros)

tempo_final = time.time()

pred = pd.DataFrame(pontos_pred, columns = ['lat', 'lon'])
pred.to_csv(path_or_buf = 'graficos_projeto_ml_puro/predicao_RFR_1000.csv')

## Plota o histograma normalizado da rodada de menor erro médio
fig, ax = plt.subplots()
ax.hist(erros, bins = 35, density = 1)
ax.set_xlabel('metros')
ax.set_ylabel('probabilidade')
ax.set_title('Histograma RFR 1000')
fig.savefig('graficos_projeto_ml_puro/Histograma_RFR_1000.png')

## Plota o boxplot de todas as rodadas
fig2, ax2 = plt.subplots()
ax2.boxplot(erros)
ax2.set_ylabel('metros')
ax2.set_title('BoxPlot RFR 1000')
fig2.savefig('graficos_projeto_ml_puro/BoxPlot_RFR_1000.png')

## Plota pontos preditos no mapa
pontos_pred_plot = pd.DataFrame(pontos_pred, columns = ['lat', 'lon'])
fig3, ax3 = plt.subplots()
ax3.scatter(y['lon'], y['lat'], s = 10, color = 'gray', alpha = 1) ## Plota todos os pontos medidos
ax3.scatter(y_test['lon'], y_test['lat'], s = 10, color = 'blue', alpha = 1) ## Plota os pontos reais de teste da rodada de menor erro medio
ax3.scatter(pontos_pred_plot['lon'], pontos_pred_plot['lat'], s = 10, color = 'green', alpha = 1) ## Plota pontos preditos da rodada de menor erro medio
ax3.scatter(ERBS_loc['lon'], ERBS_loc['lat'], s = 30, color = 'black', alpha = 1) ## Plota localidade das ERBS
ax3.set(xlabel = 'lon', ylabel = 'lat', title = 'RFR 1000 Erro: ' + str(media_erros) + ' metros')
fig3.savefig('graficos_projeto_ml_puro/Mapa_RFR_1000.png')

print('RFR 1000 | Erro medio:', np.mean(erros), '| Erro Min:', np.min(erros), '| Erro Max:', np.max(erros), '| Desvio P.:', np.std(erros)) ## Erros medio, minimo, maximo e desvio padrao da rodada de menor erro medio
print('Tempo RFR 1000:', tempo_final - tempo_inicial)

tempo_inicial = time.time()

## Para ETR 2000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 5)
etr = ExtraTreesRegressor(n_estimators = 2000)
etr.fit(X_train, y_train)
pontos_pred = etr.predict(X_test)

for i in range(150):
	erros[i] = GeoUtils.distanceInKm(y_test.iloc[i, 0], y_test.iloc[i, 1], pontos_pred[i, 0], pontos_pred[i, 1]) * 1000

media_erros = np.mean(erros)

tempo_final = time.time()

pred = pd.DataFrame(pontos_pred, columns = ['lat', 'lon'])
pred.to_csv(path_or_buf = 'graficos_projeto_ml_puro/predicao_ETR_2000.csv')

## Plota o histograma normalizado da rodada de menor erro médio
fig, ax = plt.subplots()
ax.hist(erros, bins = 35, density = 1)
ax.set_xlabel('metros')
ax.set_ylabel('probabilidade')
ax.set_title('Histograma ETR 2000')
fig.savefig('graficos_projeto_ml_puro/Histograma_ETR_2000.png')

## Plota o boxplot de todas as rodadas
fig2, ax2 = plt.subplots()
ax2.boxplot(erros)
ax2.set_ylabel('metros')
ax2.set_title('BoxPlot ETR 2000')
fig2.savefig('graficos_projeto_ml_puro/BoxPlot_ETR_2000.png')

## Plota pontos preditos no mapa
pontos_pred_plot = pd.DataFrame(pontos_pred, columns = ['lat', 'lon'])
fig3, ax3 = plt.subplots()
ax3.scatter(y['lon'], y['lat'], s = 10, color = 'gray', alpha = 1) ## Plota todos os pontos medidos
ax3.scatter(y_test['lon'], y_test['lat'], s = 10, color = 'blue', alpha = 1) ## Plota os pontos reais de teste da rodada de menor erro medio
ax3.scatter(pontos_pred_plot['lon'], pontos_pred_plot['lat'], s = 10, color = 'green', alpha = 1) ## Plota pontos preditos da rodada de menor erro medio
ax3.scatter(ERBS_loc['lon'], ERBS_loc['lat'], s = 30, color = 'black', alpha = 1) ## Plota localidade das ERBS
ax3.set(xlabel = 'lon', ylabel = 'lat', title = 'ETR 2000 Erro: ' + str(media_erros) + ' metros')
fig3.savefig('graficos_projeto_ml_puro/Mapa_ETR_2000.png')

print('ETR 2000 | Erro medio:', np.mean(erros), '| Erro Min:', np.min(erros), '| Erro Max:', np.max(erros), '| Desvio P.:', np.std(erros)) ## Erros medio, minimo, maximo e desvio padrao da rodada de menor erro medio
print('Tempo ETR 2000:', tempo_final - tempo_inicial)

tempo_inicial = time.time()

## Para KNN 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 5)
knn = KNeighborsRegressor(n_neighbors = 3)
knn.fit(X_train, y_train)
pontos_pred = knn.predict(X_test)

for i in range(150):
	erros[i] = GeoUtils.distanceInKm(y_test.iloc[i, 0], y_test.iloc[i, 1], pontos_pred[i, 0], pontos_pred[i, 1]) * 1000

media_erros = np.mean(erros)

tempo_final = time.time()

pred = pd.DataFrame(pontos_pred, columns = ['lat', 'lon'])
pred.to_csv(path_or_buf = 'graficos_projeto_ml_puro/predicao_KNN_3.csv')

## Plota o histograma normalizado da rodada de menor erro médio
fig, ax = plt.subplots()
ax.hist(erros, bins = 35, density = 1)
ax.set_xlabel('metros')
ax.set_ylabel('probabilidade')
ax.set_title('Histograma KNN 3')
fig.savefig('graficos_projeto_ml_puro/Histograma_KNN_3.png')

## Plota o boxplot de todas as rodadas
fig2, ax2 = plt.subplots()
ax2.boxplot(erros)
ax2.set_ylabel('metros')
ax2.set_title('BoxPlot KNN 3')
fig2.savefig('graficos_projeto_ml_puro/BoxPlot_KNN_3.png')

## Plota pontos preditos no mapa
pontos_pred_plot = pd.DataFrame(pontos_pred, columns = ['lat', 'lon'])
fig3, ax3 = plt.subplots()
ax3.scatter(y['lon'], y['lat'], s = 10, color = 'gray', alpha = 1) ## Plota todos os pontos medidos
ax3.scatter(y_test['lon'], y_test['lat'], s = 10, color = 'blue', alpha = 1) ## Plota os pontos reais de teste da rodada de menor erro medio
ax3.scatter(pontos_pred_plot['lon'], pontos_pred_plot['lat'], s = 10, color = 'green', alpha = 1) ## Plota pontos preditos da rodada de menor erro medio
ax3.scatter(ERBS_loc['lon'], ERBS_loc['lat'], s = 30, color = 'black', alpha = 1) ## Plota localidade das ERBS
ax3.set(xlabel = 'lon', ylabel = 'lat', title = 'KNN 3 Erro: ' + str(media_erros) + ' metros')
fig3.savefig('graficos_projeto_ml_puro/Mapa_KNN_3.png')

print('KNN 3 | Erro medio:', np.mean(erros), '| Erro Min:', np.min(erros), '| Erro Max:', np.max(erros), '| Desvio P.:', np.std(erros)) ## Erros medio, minimo, maximo e desvio padrao da rodada de menor erro medio
print('Tempo KNN 3:', tempo_final - tempo_inicial)
#plt.show()