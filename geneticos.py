import numpy as np
import matplotlib.pyplot as plt

# Função de Rastrigin
def rastrigin(X):
    A = 10
    n = len(X)
    return A * n + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

# Algoritmo Genético
class GeneticAlgorithm:
    def __init__(self, pop_size, num_generations, mutation_rate, crossover_rate, dimension, bounds):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.dimension = dimension
        self.bounds = bounds
        self.population = self.initialize_population()
        self.best_fitness_history = []  # Armazena o histórico do melhor fitness de cada geração

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dimension))

    def evaluate(self, population):
        return np.array([rastrigin(individual) for individual in population])

    def select_parents(self, fitness):
        indices = np.argsort(fitness)
        return self.population[indices[:2]]  # Seleciona os 2 melhores

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dimension - 1)
            return np.concatenate((parent1[:point], parent2[point:]))
        return parent1

    def mutate(self, individual):
        for i in range(self.dimension):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.uniform(-0.5, 0.5)
        return individual

    def evolve(self):
        for generation in range(self.num_generations):
            fitness = self.evaluate(self.population)
            new_population = []

            for _ in range(self.pop_size // 2):  # Gerar nova população
                parents = self.select_parents(fitness)
                child1 = self.mutate(self.crossover(parents[0], parents[1]))
                child2 = self.mutate(self.crossover(parents[1], parents[0]))
                new_population.extend([child1, child2])

            self.population = np.array(new_population)

            best_fitness = np.min(fitness)
            best_individual = self.population[np.argmin(fitness)]
            self.best_fitness_history.append(best_fitness)  # Armazenar o melhor valor da função
            print(f"Geração {generation}: Melhor valor da função = {best_fitness}, Melhor indivíduo = {best_individual}")

        # Após a evolução, plotar o gráfico do histórico de melhores fitness
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
mutation_rate = 0.1
crossover_rate = 0.7
dimension = 20
bounds = (-10, 10)  # Limites típicos para a função de Rastrigin

# Executar Algoritmo Genético
ga = GeneticAlgorithm(pop_size, num_generations, mutation_rate, crossover_rate, dimension, bounds)
ga.evolve()

# Após a execução, verificar se encontrou o mínimo global
minimo_global = 0  # O valor conhecido do mínimo global da função de Rastrigin é 0
melhor_valor = ga.best_fitness_history[-1]  # Pega o último melhor valor da função

if abs(melhor_valor - minimo_global) < 1e-6:
    print(f"Encontrou o mínimo global: {melhor_valor}")
else:
    print(f"Mínimo encontrado não é o global, valor encontrado: {melhor_valor}")
