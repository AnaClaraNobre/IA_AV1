import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# Função de Rastrigin
def rastrigin(x, A=10):
    p = len(x)
    return A * p + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função de Aptidão
def fitness_function(x):
    return rastrigin(x) + 1  # Ψ(x) = f(x) + 1

# Inicialização da população binária
def initialize_population_binary(pop_size, n_genes):
    return [np.random.randint(2, size=n_genes) for _ in range(pop_size)]

# Decodificação binária para valores reais
def binary_to_real(binary_individual, min_val=-10, max_val=10):
    # Converte o array binário em uma string e depois em um número decimal
    binary_str = ''.join(binary_individual.astype(str))
    decimal = int(binary_str, 2)
    # Normaliza o decimal para o intervalo [min_val, max_val]
    return min_val + (max_val - min_val) * decimal / (2**len(binary_individual) - 1)

# Inicialização da população em ponto flutuante
def initialize_population_float(pop_size, n_genes):
    return [np.random.uniform(-10, 10, n_genes) for _ in range(pop_size)]

# Avaliação da aptidão (fitness) para binário
def evaluate_population_binary(population):
    decoded_population = [binary_to_real(individual) for individual in population]
    return [fitness_function([decoded_population[i]]) for i in range(len(decoded_population))]

# Avaliação da aptidão (fitness) para ponto flutuante
def evaluate_population_float(population):
    return [fitness_function(individual) for individual in population]

# Seleção por roleta
def roulette_wheel_selection(population, fitness):
    total_fitness = sum(fitness)
    probs = [f / total_fitness for f in fitness]
    selected = []
    for _ in range(len(population)):
        selected.append(population[np.random.choice(range(len(population)), p=probs)])
    return selected

# Crossover de um ponto para binário
def crossover_binary(parent1, parent2):
    if random.random() < 0.9:  # 90% de taxa de crossover
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1, parent2

# Mutação binária
def mutate_binary(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Mutação binária
    return individual

# Algoritmo Genético 1 (Binário)
def genetic_algorithm_binary(pop_size=100, n_genes=20, n_generations=1000, mutation_rate=0.01, tol=1e-6):
    population = initialize_population_binary(pop_size, n_genes)
    best_fitness = float('inf')
    
    for generation in range(n_generations):
        # Avaliar população
        fitness = evaluate_population_binary(population)
        
        # Verificar convergência
        current_best_fitness = min(fitness)
        if abs(best_fitness - current_best_fitness) < tol:
            break
        best_fitness = current_best_fitness
        
        # Seleção por roleta
        selected = roulette_wheel_selection(population, fitness)
        
        # Gerar nova população com crossover e mutação
        next_population = []
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[i + 1]
            child1, child2 = crossover_binary(parent1, parent2)
            next_population.append(mutate_binary(child1, mutation_rate))
            next_population.append(mutate_binary(child2, mutation_rate))
        
        population = next_population
        
    # Decodificar a população final e obter o melhor indivíduo
    decoded_population = [binary_to_real(individual) for individual in population]
    fitness = evaluate_population_binary(population)
    best_individual = decoded_population[np.argmin(fitness)]
    return best_individual, best_fitness

# Simulated Binary Crossover (SBX) para ponto flutuante
def sbx_crossover(parent1, parent2, eta=1.0):
    child1, child2 = np.copy(parent1), np.copy(parent2)
    if random.random() < 0.9:  # Taxa de crossover > 85%
        for i in range(len(parent1)):
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
    return child1, child2

# Mutação Gaussiana para ponto flutuante
def gaussian_mutation(individual, mutation_rate=0.01, sigma=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += np.random.normal(0, sigma)  # Mutação Gaussiana
    return individual

# Seleção por Torneio
def tournament_selection(population, fitness, k=3):
    selected = []
    for _ in range(len(population)):
        aspirants = random.sample(list(zip(population, fitness)), k)
        selected.append(min(aspirants, key=lambda x: x[1])[0])
    return selected

# Algoritmo Genético 2 (Ponto Flutuante)
def genetic_algorithm_float(pop_size=100, n_genes=20, n_generations=1000, mutation_rate=0.01, tol=1e-6):
    population = initialize_population_float(pop_size, n_genes)
    best_fitness = float('inf')
    
    for generation in range(n_generations):
        # Avaliar população
        fitness = evaluate_population_float(population)
        
        # Verificar convergência
        current_best_fitness = min(fitness)
        if abs(best_fitness - current_best_fitness) < tol:
            break
        best_fitness = current_best_fitness
        
        # Seleção por torneio
        selected = tournament_selection(population, fitness)
        
        # Gerar nova população com crossover SBX e mutação Gaussiana
        next_population = []
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[i + 1]
            child1, child2 = sbx_crossover(parent1, parent2)
            next_population.append(gaussian_mutation(child1, mutation_rate))
            next_population.append(gaussian_mutation(child2, mutation_rate))
        
        population = next_population
        
    best_individual = population[np.argmin(fitness)]
    return best_individual, best_fitness

# Função para rodar várias execuções do algoritmo genético e coletar as estatísticas
def run_multiple_executions(algorithm_func, n_executions=100):
    results = []
    
    for _ in range(n_executions):
        _, best_fitness = algorithm_func()
        results.append(best_fitness)
    
    # Calcular estatísticas
    min_fitness = np.min(results)
    max_fitness = np.max(results)
    mean_fitness = np.mean(results)
    std_fitness = np.std(results)
    
    return min_fitness, max_fitness, mean_fitness, std_fitness

# Executar 100 vezes para o algoritmo binário
min_fit_1, max_fit_1, mean_fit_1, std_fit_1 = run_multiple_executions(genetic_algorithm_binary)

# Executar 100 vezes para o algoritmo ponto flutuante
min_fit_2, max_fit_2, mean_fit_2, std_fit_2 = run_multiple_executions(genetic_algorithm_float)

# Exibir tabela comparativa
# print("Comparação dos Algoritmos Genéticos:")
# print(f"{'Métrica':<20} {'Algoritmo 1 (Binário)':<20} {'Algoritmo 2 (Flutuante)':<20}")
# print(f"{'Menor valor':<20} {min_fit_1:<20} {min_fit_2:<20}")
# print(f"{'Maior valor':<20} {max_fit_1:<20} {max_fit_2:<20}")
# print(f"{'Média':<20} {mean_fit_1:<20} {mean_fit_2:<20}")
# print(f"{'Desvio Padrão':<20} {std_fit_1:<20} {std_fit_2:<20}")

# Criar DataFrame com os resultados
results_df = pd.DataFrame({
    'Métrica': ['Menor valor', 'Maior valor', 'Média', 'Desvio Padrão'],
    'Algoritmo 1 (Binário)': [min_fit_1, max_fit_1, mean_fit_1, std_fit_1],
    'Algoritmo 2 (Flutuante)': [min_fit_2, max_fit_2, mean_fit_2, std_fit_2]
})

# Exibir a tabela comparativa
print("Comparação dos Algoritmos Genéticos:\n")
print(results_df)

# Plotar gráficos
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Gráfico de Linhas para métricas
metrics = ['Menor valor', 'Maior valor', 'Média', 'Desvio Padrão']
bin_values = [min_fit_1, max_fit_1, mean_fit_1, std_fit_1]
float_values = [min_fit_2, max_fit_2, mean_fit_2, std_fit_2]

ax.plot(metrics, bin_values, marker='o', linestyle='-', color='b', label='Algoritmo 1 (Binário)')
ax.plot(metrics, float_values, marker='o', linestyle='-', color='r', label='Algoritmo 2 (Flutuante)')
ax.set_xlabel('Métricas')
ax.set_ylabel('Valor')
ax.set_title('Comparação dos Algoritmos Genéticos')
ax.legend()

plt.tight_layout()
plt.show()