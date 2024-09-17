import numpy as np
import matplotlib.pyplot as plt

# Função de Rastrigin com constante A
def rastrigin(X):
    A = 10
    p = len(X)
    return A * p + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])

# Função de Aptidão Ψ(x) = f (x) + 1.
def fitness_function(X):
    return rastrigin(X) + 1

# Algoritmo Genético com Representação Cromossômica em Ponto Flutuante
class GeneticAlgorithmFloatSBX:
    def __init__(self, pop_size, num_generations, mutation_rate, crossover_rate, dimension, eta_c, bounds, tournament_size):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.dimension = dimension
        self.bounds = bounds
        self.eta_c = eta_c
        self.tournament_size = tournament_size
        self.population = self.initialize_population()
        self.best_fitness_history = []   # Armazena o histórico do melhor fitness de cada geração

    def initialize_population(self):
        # Inicializar população com números reais entre os limites
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dimension))

    def evaluate(self, population):
        return np.array([fitness_function(individual) for individual in population])
    
        # Crossover Simulated Binary (SBX)
    def crossover_sbx(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
            for i in range(self.dimension):
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u)**(1 / (self.eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u)))**(1 / (self.eta_c + 1))

                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

                # Garantir que os valores das crianças estão dentro dos limites
                child1[i] = np.clip(child1[i], self.bounds[0], self.bounds[1])
                child2[i] = np.clip(child2[i], self.bounds[0], self.bounds[1])

            return child1, child2
        return parent1, parent2

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            # Realizar crossover de ponto único em ponto flutuante
            point = np.random.randint(1, self.dimension - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        for i in range(self.dimension):
            if np.random.rand() < self.mutation_rate:
                # Realizar mutação, alterando ligeiramente o valor da variável
                individual[i] += np.random.uniform(-0.1, 0.1)
                # Garantir que o valor da variável permaneça dentro dos limites
                individual[i] = np.clip(individual[i], self.bounds[0], self.bounds[1])
        return individual

    def select_parents(self, fitness):
        tournament_indices = np.random.choice(np.arange(self.pop_size),size=tournament_size)
        tournament_fitness = fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return self.population[winner_idx]
    
    # Critério de parada baseado no número de gerações sem melhoria
    def evolve(self):
        no_improvement_count = 0
        max_no_improvement = 50

        for generation in range(self.num_generations):
            fitness = self.evaluate(self.population)
            new_population = []

            for _ in range(self.pop_size // 2):  # Gerar nova população
                parent1 = self.select_parents(fitness)
                parent2 = self.select_parents(fitness)
                child1, child2 = self.crossover_sbx(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])

            self.population = np.array(new_population)

            best_fitness = np.min(fitness)
            self.best_fitness_history.append(best_fitness)  # Armazenar o melhor valor da função

            if generation > 0 and best_fitness >= self.best_fitness_history[-2]:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    print(f"Convergência detectada na geração {generation}.")
                    break
            else:
                no_improvement_count = 0

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
eta_c = 2
tournament_size = 3

# Executar Algoritmo Genético com Representação em Ponto Flutuante
# ga_float = GeneticAlgorithmFloat(pop_size, num_generations, mutation_rate, crossover_rate, dimension, bounds)
# ga_float.evolve()
ga_float_sbx = GeneticAlgorithmFloatSBX(pop_size, num_generations, mutation_rate, crossover_rate, dimension, eta_c, bounds, tournament_size)
ga_float_sbx.evolve()

# Verificar se encontrou o mínimo global
# minimo_global = 0  # O valor conhecido do mínimo global da função de Rastrigin é 0
# melhor_valor = ga_float.best_fitness_history[-1]  # Pega o último melhor valor da função
minimo_global = 0  # O valor conhecido do mínimo global da função de Rastrigin é 0
melhor_valor = ga_float_sbx.best_fitness_history[-1]  # Pega o último melhor valor da função


if abs(melhor_valor - minimo_global) < 1e-6:
    print(f"Encontrou o mínimo global: {melhor_valor}")
else:
    print(f"Mínimo encontrado não é o global, valor encontrado: {melhor_valor}")
