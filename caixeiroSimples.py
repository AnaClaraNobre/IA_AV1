import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# Defina o caminho do arquivo CSV
caminho_arquivo = '/Users/angelobarcelos/Documents/ACADEMIC/Unifor/IA/CaixeiroSimples.csv'

# Leia o arquivo CSV
tabela = pd.read_csv(caminho_arquivo, header=None, names=['X', 'Y', 'Z'])

# Converta a tabela em um array NumPy
pontos = tabela.to_numpy()

# Função para calcular a distância euclidiana entre dois pontos
def distancia(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Função para calcular o comprimento total do caminho
def comprimento_passo(caminho):
    comprimento = 0
    for i in range(len(caminho) - 1):
        comprimento += distancia(caminho[i], caminho[i + 1])
    # Retornar à origem
    comprimento += distancia(caminho[-1], caminho[0])
    return comprimento

# Função de aptidão (minimizar o comprimento do caminho)
def avaliar_individuo(individuo):
    caminho = [pontos[i] for i in individuo]
    comprimento = comprimento_passo(caminho)
    return (1 / comprimento, )  # Retorna uma tupla

# Configurar o Algoritmo Genético
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(pontos)), len(pontos))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", avaliar_individuo)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Parâmetros do Algoritmo Genético
population = toolbox.population(n=100)
NGEN = 500
CXPB, MUTPB = 0.7, 0.2

# Executar o Algoritmo Genético
for gen in range(NGEN):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = toolbox.select(population + offspring, len(population))

# Extrair o melhor indivíduo
best_ind = tools.selBest(population, 1)[0]
melhor_caminho = [pontos[i] for i in best_ind]

# Separar as coordenadas X, Y e Z do melhor caminho
melhor_caminho = np.array(melhor_caminho)
x = melhor_caminho[:, 0]
y = melhor_caminho[:, 1]
z = melhor_caminho[:, 2]

# Plotar o gráfico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Caminho do Caixeiro Viajante em 3D com Algoritmo Genético')

plt.show()