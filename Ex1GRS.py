import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Função a ser minimizada
def objective_function(x):
    return x[0]**2 + x[1]**2

# Função para mostrar o gráfico ao encontrar um novo valor mínimo
def mostrar_grafico_grs(x_opt, f_opt):
    x1_range = np.linspace(-100, 100, 400)
    x2_range = np.linspace(-100, 100, 400)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_grid = objective_function([x1_grid, x2_grid])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, z_grid, cmap='viridis', alpha=0.6)
    ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
    ax.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
    ax.set_zticks([2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
    plt.title(f'Novo valor mínimo: f(x1, x2) = {f_opt:.6f}')
    plt.show()

# Ajustar o Global Random Search (GRS)
def global_random_search_with_plot(n_iterations, bounds):
    best = np.random.uniform(low=bounds[0], high=bounds[1], size=2)
    best_eval = objective_function(best)

    for i in range(n_iterations):
        candidate = np.random.uniform(low=bounds[0], high=bounds[1], size=2)
        candidate_eval = objective_function(candidate)

        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval

    return best, best_eval

# Parâmetros
R = 100
n_iterations = 10000
bounds = [-100, 100]
melhor_solucao_grs = None
melhor_valor_grs = float('inf')
resultados_grs = []
tempo_total_inicio = time.time()

# Executar o algoritmo Global Random Search (GRS)
for r in range(R):
    print(f"\nExecução {r + 1}/{R}:")
    x_result_grs, f_result_grs = global_random_search_with_plot(n_iterations, bounds)

    resultados_grs.append(f_result_grs)
    
    print(f"Rodada {r + 1}: Solução = {x_result_grs}, f(x) = {f_result_grs:.6f}")

    if f_result_grs < melhor_valor_grs:
        melhor_valor_grs = f_result_grs
        melhor_solucao_grs = x_result_grs

tempo_total_fim = time.time()
tempo_total = tempo_total_fim - tempo_total_inicio

# Exibir a melhor solução e o resultado mais frequente
print(f"\nMelhor solução encontrada: x1 = {melhor_solucao_grs[0]:.6f}, x2 = {melhor_solucao_grs[1]:.6f}")
print(f"Valor mínimo da função: {melhor_valor_grs:.6f}")

# Mostrar o gráfico da melhor solução encontrada
mostrar_grafico_grs(melhor_solucao_grs, melhor_valor_grs)

resultado_mais_frequente_grs = Counter(resultados_grs).most_common(1)[0]
print(f"O resultado mais frequente foi {resultado_mais_frequente_grs[0]:.6f}, encontrado {resultado_mais_frequente_grs[1]} vezes.")
print(f"\nTempo total de execução: {tempo_total:.4f} segundos")
