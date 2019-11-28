import pandas as pd
import numpy as np

notas = [8.5, 7, 9]
alunos = ['A', 'B', 'C']
#alunos_notas = pd.Series({'A': 8.5, 'B': 7, 'C': 9})
#alunos_notas = pd.Series(data = [8.5, 7, 9], index = ['A', 'B', 'C'])
alunos_notas = pd.Series(notas, alunos)  ## indice --> alunos
alunos_notas += 1
print(alunos_notas)
print()
print(alunos_notas['B'])
print()
print(alunos_notas[['A', 'C', 'B']])
print()


###NaN#####
series_e = pd.Series(data = [5, 9, 10, 5], index = ['AL01', 'AL02', 'AL03', 'AL04'])
series_f = pd.Series(data = [4, 6, 4], index = ['AL01', 'AL02', 'AL03'])
sum_e_f = series_e + series_f
print(sum_e_f)
print()

#####Metodos#######
print(series_e)
print()
print(series_e.max())
print()
print(series_e.min())
print()
print(series_e.describe())
print()
print(series_e.cummax())   ## maximo ate certo ponto
print()
print(series_e.cumprod())  ## produtorio
print()
print(series_e.cumsum())  ## somatorio
print()

###########DataFrames###########################
#########Criar a partir de uma matriz############
dados = np.random.rand(5, 4) * 10
print('Dados: ')
print(dados)
print()
index_linha = ['ST01','ST02','ST03','ST04','ST05']
index_coluna = ['T1', 'T2', 'T3', '4T']
df = pd.DataFrame(dados, index_linha, index_coluna)
print('DataFrame: ')
print(df)
print()
print('Coluna T1: ')
print(df['T1'])
print()
print('Colunas T1 e 4T: ')
print(df[['T1', '4T']])
print()

###########Adicionar Colunas#################
df['T4'] = np.random.rand(5) * 10
print('DataFrame: ')
print(df)
print()

######operacoes##############
df['AVG'] = (df['T1'] + df['T2'] + df['T3'] + df['T4'] + df['4T']) / 5
print('DataFrame: ')
print(df)
print()

df['AVG'] = df.mean(1)
print('DataFrame: ')
print(df)
print()

######Remover colunas#########################
df2 = df.drop(['AVG', '4T'], axis = 1)
df3 = df.drop(['ST05', 'ST01'], axis = 0)
print('DataFrame2: ')
print(df2)
print()
print('DataFrame3: ')
print(df3)
print()

################Metodos#####################
df3 = df.loc['ST05']
print('DataFrame3: ')
print(df3)
print()
df3 = df.iloc[4]
print('DataFrame3: ')
print(df3)
print()

df3 = df.loc['ST05']['T1']
print('DataFrame3: ')
print(df3)
print()
df3 = df.iloc[4]['T1']
print('DataFrame3: ')
print(df3)
print()

df3 = df.loc[['ST04', 'ST05']][['T1', 'T4']]
print('DataFrame3: ')
print(df3)
print()

df3 = df.iloc[[3, 4], [0, 4]]  ## Notacao de matrizes, diferente da de cima
print('DataFrame3: ')
print(df3)
print()

df3 = df.loc[['ST05', 'ST01']]
print('DataFrame3: ')
print(df3)
print()

print('DataFrame: ')
print(df)
print()

df = df.drop(['4T', 'AVG'], axis = 1)
print(df)
print()

df = df.reset_index(drop = False) ## apaga a coluna de indices se verdadeiro, se falso mantem a coluna mas com indices default
print(df)
print()

novos_indices = ['ST06','ST07','ST08','ST09','ST010']
df['STs'] = novos_indices
print(df)
print()

df = df.set_index('STs', drop = True) ## cria uma duplicata da coluna STs e seta como indice se falso e mantem a coluna original, 
print(df)  						      ## se verdadeiro apaga a coluna original
print()

df.rename(columns = {'index': 'STsAntigos'}, inplace = True)
print(df)
print()

#############Limpando DataFrame######################
dict_a = {'X':[1, np.nan, np.nan],'Y':[2, 4, np.nan],'Z':[3, 4, 4],}
df3 = pd.DataFrame(dict_a)
print(df3)
print()

df4 = df3.dropna()
print(df4)
print()

df5 = df3.dropna(axis = 1)
print(df5)
print()

df6 = df3.dropna(axis = 1, thresh = 2)
print(df6)
print()

df7 = df3.fillna(3)
print(df7)
print()