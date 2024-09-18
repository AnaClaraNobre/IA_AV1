# import numpy as np
# import random
# import pandas as pd

# # Função para calcular a distância euclidiana entre dois pontos
# def euclidean_distance(p1, p2):
#     return np.sqrt(np.sum((p1 - p2) ** 2))

# # Função para calcular a distância total de uma rota
# def total_distance(route, points):
#     distance = 0
#     num_points = len(route)
#     for i in range(num_points):
#         distance += euclidean_distance(points[route[i]], points[route[ (i + 1) % num_points]])
#     return distance

# # Função de aptidão: inversamente proporcional à distância total
# def fitness(route, points):
#     return 1 / total_distance(route, points)

# # Operador de seleção por torneio
# def tournament_selection(population, fitness_scores, k):
#     selected = random.sample(range(len(population)), k)
#     best = max(selected, key=lambda x: fitness_scores[x])
#     return population[best]

# # Operador de recombinação OX (Order Crossover)
# def order_crossover(parent1, parent2):
#     size = len(parent1)
#     start, end = sorted(random.sample(range(size), 2))
    
#     offspring = [None] * size
#     offspring[start:end] = parent1[start:end]
    
#     current_pos = end
#     for gene in parent2:
#         if gene not in offspring:
#             if current_pos >= size:
#                 current_pos = 0
#             offspring[current_pos] = gene
#             current_pos += 1
            
#     return offspring

# # Operador de mutação por troca de dois pontos
# def swap_mutation(route, mutation_rate):
#     if random.random() < mutation_rate:
#         idx1, idx2 = random.sample(range(len(route)), 2)
#         route[idx1], route[idx2] = route[idx2], route[idx1]

# # Função principal do algoritmo genético
# def genetic_algorithm(points, population_size, max_generations, mutation_rate, num_elites):
#     num_points = len(points)
    
#     # Inicialização da população
#     population = [random.sample(range(num_points), num_points) for _ in range(population_size)]
    
#     for generation in range(max_generations):
#         fitness_scores = [fitness(route, points) for route in population]
        
#         # Seleção dos elites
#         sorted_population = [route for _, route in sorted(zip(fitness_scores, population), reverse=True)]
#         elites = sorted_population[:num_elites]
        
#         # Criação da nova população
#         new_population = elites.copy()
        
#         while len(new_population) < population_size:
#             parent1 = tournament_selection(population, fitness_scores, k=3)
#             parent2 = tournament_selection(population, fitness_scores, k=3)
            
#             offspring = order_crossover(parent1, parent2)
#             swap_mutation(offspring, mutation_rate)
            
#             new_population.append(offspring)
        
#         population = new_population
        
#     # Melhor solução final
#     best_route = max(population, key=lambda r: fitness(r, points))
#     return fitness(best_route, points)

# # Função para executar múltiplas rodadas e coletar estatísticas
# def run_multiple_trials(points, num_trials, population_size, max_generations, mutation_rate, num_elites):
#     fitness_results = []
    
#     for _ in range(num_trials):
#         best_fitness = genetic_algorithm(points, population_size, max_generations, mutation_rate, num_elites)
#         fitness_results.append(best_fitness)
    
#     # Cálculo das estatísticas
#     min_fitness = min(fitness_results)
#     max_fitness = max(fitness_results)
#     mean_fitness = np.mean(fitness_results)
#     std_dev_fitness = np.std(fitness_results)
    
#     # Criação do DataFrame com pandas
#     data = {
#         "Menor Valor de Aptidão": [min_fitness],
#         "Maior Valor de Aptidão": [max_fitness],
#         "Média de Valor de Aptidão": [mean_fitness],
#         "Desvio-Padrão de Aptidão": [std_dev_fitness]
#     }
#     df = pd.DataFrame(data)
    
#     return df

# # Definições do problema
# points = np.array([
#     [0, 0],
#     [1, 2],
#     [4, 5],
#     [7, 8],
#     [10, 11],
#     [13, 14]
# ])

# num_trials = 100
# population_size = 100
# max_generations = 500
# mutation_rate = 0.01
# num_elites = 5

# # Execução das múltiplas rodadas
# results_df = run_multiple_trials(
#     points, num_trials, population_size, max_generations, mutation_rate, num_elites
# )

# # Exibição dos resultados
# print(results_df)
import numpy as np
import random
import pandas as pd

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Função para calcular a distância total de uma rota
def total_distance(route, points):
    distance = 0
    num_points = len(route)
    for i in range(num_points):
        distance += euclidean_distance(points[route[i]], points[route[(i + 1) % num_points]])
    return distance

# Função de aptidão: inversamente proporcional à distância total
def fitness(route, points):
    return 1 / total_distance(route, points)

# Operador de seleção por torneio
def tournament_selection(population, fitness_scores, k):
    selected = random.sample(range(len(population)), k)
    best = max(selected, key=lambda x: fitness_scores[x])
    return population[best]

# Operador de recombinação OX (Order Crossover)
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    offspring = [None] * size
    offspring[start:end] = parent1[start:end]
    
    current_pos = end
    for gene in parent2:
        if gene not in offspring:
            if current_pos >= size:
                current_pos = 0
            offspring[current_pos] = gene
            current_pos += 1
            
    return offspring

# Operador de mutação por troca de dois pontos
def swap_mutation(route, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]

# Função para verificar a validade da rota
def is_valid_route(route):
    return len(route) == len(set(route))

# Função principal do algoritmo genético
def genetic_algorithm(points, population_size, max_generations, mutation_rate, num_elites):
    num_points = len(points)
    
    # Inicialização da população
    population = [random.sample(range(num_points), num_points) for _ in range(population_size)]
    
    for generation in range(max_generations):
        fitness_scores = [fitness(route, points) for route in population]
        
        # Verificação de validade das rotas na população
        for route in population:
            assert is_valid_route(route), "Rota inválida com pontos repetidos!"
        
        # Seleção dos elites
        sorted_population = [route for _, route in sorted(zip(fitness_scores, population), reverse=True)]
        elites = sorted_population[:num_elites]
        
        # Criação da nova população
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores, k=3)
            parent2 = tournament_selection(population, fitness_scores, k=3)
            
            offspring = order_crossover(parent1, parent2)
            swap_mutation(offspring, mutation_rate)
            
            # Verificação de validade da nova rota gerada
            assert is_valid_route(offspring), "Rota inválida com pontos repetidos!"
            
            new_population.append(offspring)
        
        population = new_population
        
    # Melhor solução final
    best_route = max(population, key=lambda r: fitness(r, points))
    return fitness(best_route, points)

# Função para executar múltiplas rodadas e coletar estatísticas
def run_multiple_trials(points, num_trials, population_size, max_generations, mutation_rate, num_elites):
    fitness_results = []
    
    for _ in range(num_trials):
        best_fitness = genetic_algorithm(points, population_size, max_generations, mutation_rate, num_elites)
        fitness_results.append(best_fitness)
    
    # Cálculo das estatísticas
    min_fitness = min(fitness_results)
    max_fitness = max(fitness_results)
    mean_fitness = np.mean(fitness_results)
    std_dev_fitness = np.std(fitness_results)
    
    # Criação do DataFrame com pandas
    data = {
        "Menor Valor de Aptidão": [min_fitness],
        "Maior Valor de Aptidão": [max_fitness],
        "Média de Valor de Aptidão": [mean_fitness],
        "Desvio-Padrão de Aptidão": [std_dev_fitness]
    }
    df = pd.DataFrame(data)
    
    return df

# Definições do problema
points = np.array([
    [0, 0],
    [1, 2],
    [4, 5],
    [7, 8],
    [10, 11],
    [13, 14]
])

num_trials = 100
population_size = 100
max_generations = 500
mutation_rate = 0.01
num_elites = 5

# Execução das múltiplas rodadas
results_df = run_multiple_trials(
    points, num_trials, population_size, max_generations, mutation_rate, num_elites
)

# Exibição dos resultados
print(results_df)
