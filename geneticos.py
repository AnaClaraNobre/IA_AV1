import numpy as np
import matplotlib.pyplot as plt

# Função para converter binário para número real
def bin_to_real(binary, bounds, bits_per_variable):
    lower_bound, upper_bound = bounds
    decimal = int(binary, 2)
    max_decimal = 2**bits_per_variable - 1
    return lower_bound + (upper_bound - lower_bound) * decimal / max_decimal

# Função para converter real para binário
def real_to_bin(value, bounds, bits_per_variable):
    lower_bound, upper_bound = bounds
    decimal = int((value - lower_bound) / (upper_bound - lower_bound) * (2**bits_per_variable - 1))
    return format(decimal, f'0{bits_per_variable}b')

# Função de Rastrigin com constante A
def rastrigin(X):
    A = 10
    p = len(X)
    return A * p + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

#Função de Aptidão Ψ(x) = f (x) + 1.
def fitness_function(X):
    return rastrigin(X) + 1

# Algoritmo Genético com Representação Cromossômica Canônica
class GeneticAlgorithmBinary:
    def __init__(self, pop_size, num_generations, mutation_rate, crossover_rate, dimension, bounds, bits_per_variable):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.dimension = dimension
        self.bounds = bounds
        self.bits_per_variable = bits_per_variable
        self.population = self.initialize_population()
        self.best_fitness_history = []  # Armazena o histórico do melhor fitness de cada geração

    def initialize_population(self):
        # Inicializar população como strings binárias
        return [''.join(np.random.choice(['0', '1'], self.bits_per_variable * self.dimension)) for _ in range(self.pop_size)]

    def evaluate(self, population):
        real_population = [self.decode_chromosome(chromosome) for chromosome in population]
        return np.array([fitness_function(individual) for individual in real_population])

    def decode_chromosome(self, chromosome):
        # Dividir o cromossomo binário em segmentos para cada variável
        real_values = []
        for i in range(self.dimension):
            start = i * self.bits_per_variable
            end = start + self.bits_per_variable
            binary_segment = chromosome[start:end]
            real_value = bin_to_real(binary_segment, self.bounds, self.bits_per_variable)
            real_values.append(real_value)
        return real_values

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.bits_per_variable * self.dimension - 1)
            return parent1[:point] + parent2[point:]
        return parent1

    def mutate(self, chromosome):
        chromosome = list(chromosome)  # Convert to list for easier mutation
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = '1' if chromosome[i] == '0' else '0'
        return ''.join(chromosome)
    
    def select_parents(self, fitness):
        total_fitness = np.sum(fitness)
        probabilities = fitness/total_fitness

        selected_indices = np.random.choice(np.arange(self.pop_size),size=2,p=probabilities)
        return [self.population[idx] for idx in selected_indices]
    
    def evolve(self):
        for generation in range(self.num_generations):
            fitness = self.evaluate(self.population)
            new_population = []

            for _ in range(self.pop_size // 2):  # Gerar nova população
                parents = self.select_parents(fitness)
                child1 = self.mutate(self.crossover(parents[0], parents[1]))
                child2 = self.mutate(self.crossover(parents[1], parents[0]))
                new_population.extend([child1, child2])

            self.population = new_population

            best_fitness = np.min(fitness)
            best_individual = self.population[np.argmin(fitness)]
            self.best_fitness_history.append(best_fitness)  # Armazenar o melhor valor da função
            print(f"Geração {generation}: Melhor valor da função = {best_fitness}")

        self.plot_fitness_history()

    def plot_fitness_history(self):
        # Plotar o histórico de fitness
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='Melhor Fitness (mínimo)')
        plt.title('Evolução do Melhor Fitness ao Longo das Gerações')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True)
        plt.legend()
        plt.show()

# Parâmetros
pop_size = 100
num_generations = 200
mutation_rate = 0.01
crossover_rate = 0.9
dimension = 20  # p = 20
bounds = (-10, 10)  # Limites de restrição [-10, 10]
bits_per_variable = 16  # Número de bits para codificar cada variável

# Executar Algoritmo Genético
ga_binary = GeneticAlgorithmBinary(pop_size, num_generations, mutation_rate, crossover_rate, dimension, bounds, bits_per_variable)
ga_binary.evolve()

# Verificar se encontrou o mínimo global
minimo_global = 0  # O valor conhecido do mínimo global da função de Rastrigin é 0
melhor_valor = ga_binary.best_fitness_history[-1]  # Pega o último melhor valor da função

if abs(melhor_valor - minimo_global) < 1e-6:
    print(f"Encontrou o mínimo global: {melhor_valor}")
else:
    print(f"Mínimo encontrado não é o global, valor encontrado: {melhor_valor}")