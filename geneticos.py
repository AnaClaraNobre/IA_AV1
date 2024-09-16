import random
import numpy as np
from deap import base, creator, tools, algorithms

# Parâmetros
A = 10
P = 20  # Dimensão do problema
LIMIT = 100  # Número de gerações
BOUND_LOW, BOUND_UP = -10.0, 10.0

# Função de Rastrigin
def rastrigin(individual):
    # Verificar se os valores do indivíduo são reais
    assert all(isinstance(x, float) for x in individual), "Indivíduo contém valores não reais!"
    return A * P + sum((x ** 2 - A * np.cos(2 * np.pi * x)) for x in individual),

# Criando as classes para a função de aptidão e o indivíduo
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimização
creator.create("Individual", list, fitness=creator.FitnessMin)

# Função para criar um indivíduo
def create_individual():
    # Garantir que os indivíduos são gerados corretamente dentro dos limites
    individual = [random.uniform(BOUND_LOW, BOUND_UP) for _ in range(P)]
    return creator.Individual(individual)

# Função para criar a população
def create_population(n):
    return [create_individual() for _ in range(n)]

# Primeiro Algoritmo Genético
def run_genetic_algorithm_1():
    toolbox = base.Toolbox()
    
    # Registro das operações
    toolbox.register("individual", create_individual)
    toolbox.register("population", create_population, n=300)
    toolbox.register("evaluate", rastrigin)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Recombinacao blend crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Mutacao Gaussiana
    toolbox.register("select", tools.selRoulette)  # Seleção por roleta
    toolbox.register("map", map)
    
    # Definindo a população
    pop = toolbox.population()
    
    # Estatísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("mean", np.mean)
    stats.register("std", np.std)
    
    # Executando o algoritmo
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.85, mutpb=0.2, ngen=LIMIT, 
                                       stats=stats, verbose=True)
    
    return pop, logbook

# Segundo Algoritmo Genético
def run_genetic_algorithm_2():
    toolbox = base.Toolbox()
    
    # Registro das operações
    toolbox.register("individual", create_individual)
    toolbox.register("population", create_population, n=300)
    toolbox.register("evaluate", rastrigin)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=10.0)  # Crossover SBX
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Mutacao Gaussiana
    toolbox.register("select", tools.selTournament, tournsize=3)  # Seleção por torneio
    toolbox.register("map", map)
    
    # Definindo a população
    pop = toolbox.population()
    
    # Estatísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("mean", np.mean)
    stats.register("std", np.std)
    
    # Executando o algoritmo
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.85, mutpb=0.2, ngen=LIMIT, 
                                       stats=stats, verbose=True)
    
    return pop, logbook

# Função para coletar estatísticas de 100 rodadas
def collect_statistics(runs=100):
    stats_1 = []
    stats_2 = []
    
    for _ in range(runs):
        _, logbook_1 = run_genetic_algorithm_1()
        _, logbook_2 = run_genetic_algorithm_2()
        
        min_fit_1 = logbook_1.select("min")[-1]
        max_fit_1 = logbook_1.select("max")[-1]
        mean_fit_1 = logbook_1.select("mean")[-1]
        std_fit_1 = logbook_1.select("std")[-1]
        
        min_fit_2 = logbook_2.select("min")[-1]
        max_fit_2 = logbook_2.select("max")[-1]
        mean_fit_2 = logbook_2.select("mean")[-1]
        std_fit_2 = logbook_2.select("std")[-1]
        
        stats_1.append((min_fit_1, max_fit_1, mean_fit_1, std_fit_1))
        stats_2.append((min_fit_2, max_fit_2, mean_fit_2, std_fit_2))
    
    return stats_1, stats_2

# Execução dos algoritmos e coleta de estatísticas
stats_algo1, stats_algo2 = collect_statistics()

# Exibição dos resultados
print("Resultados do Algoritmo Genético 1:")
print("Mínimo, Máximo, Média, Desvio Padrão")
for stats in stats_algo1:
    print(stats)

print("\nResultados do Algoritmo Genético 2:")
print("Mínimo, Máximo, Média, Desvio Padrão")
for stats in stats_algo2:
    print(stats)