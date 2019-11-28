import numpy as np


########Lista para Vetor########
lista = [1, 2, 3, 4]
vetor = np.array(lista)
print (lista)
print (vetor)
print ()

#######Lista de listas para matriz##########
lista_de_listas = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matriz = np.array(lista_de_listas)
print (lista_de_listas)
print (matriz)
print ()

#######Usando range para gerar vetor##########
array1 = np.arange(10)
print (array1)
array2 = np.arange(10, 22, 2)
print (array2)
array3 = np.arange(22, 10, -2)
print (array3)
print ()

#######Matriz identidade################
identidade = np.eye(3)
print (identidade)
print ()

######Matriz ou vetor de zeros##############
zeros = np.zeros((3, 4))
print (zeros)
print ()

########Matriz ou vetor de uns##############
uns = np.ones((3, 4))
print (uns)
print ()

#########Vetor de inteiros aleatorios##########
aleatorios = np.random.randint(1, 21, 5)
print (aleatorios)
print ()

########Vetor de numeros aleatorios com distribuicao uniforme#########
unif = np.random.rand(5)
print (unif)
print ()

########Vetor de numeros aleatorios com distribuicao normal#########
normal = np.random.randn(5)
print (normal)
print ('MAX: ' + str(normal.max()) + ', INDEX: ' + str(normal.argmax()))
print ()

