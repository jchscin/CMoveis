import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = load_boston()
#print(boston.DESCR)
#print(type(boston.data))

df = pd.DataFrame(boston.data)
'''print(df)
print()
'''
df['PRICE'] = boston.target
print(df)
print()

'''sns.pairplot(df)
plt.show()'''

'''sns.distplot(df['PRICE'])
plt.show()'''

'''print(df.corr())
sns.heatmap(df.corr())  ## plota heatmap sem valores
plt.show()'''

'''plt.rc("figure", figsize = (12, 8))
sns.heatmap(df.corr(), annot = True)  ## plota heatmap com valores
plt.show()'''

############Entrando em modelos de regressao################
X = df.drop('PRICE', axis = 1)
y = df['PRICE']
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)   ## random_state = z garante que este mesmo conjunto de dados rodado uma vez sera dividido igual sempre
reg1 = LinearRegression()
reg1.fit(X_train, y_train)

y_pred = reg1.predict(X_test)
'''
figure, axes = plt.subplots()
axes.scatter(y_pred, y_test)
axes.set_xlabel("Predicted")
axes.set_ylabel("Real")
axes.set_xlim([0, 60])
axes.set_ylim([0, 60])

sns.scatterplot(y_pred, y_test)
'''
sns.distplot(y_pred)
sns.distplot(y_test)
plt.show()


