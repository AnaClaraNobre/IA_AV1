import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Função a ser minimizada
def objective_function(x):
    return x[0]**2 + x[1]**2

# Função para mostrar o gráfico ao encontrar um novo valor mínimo
def mostrar_grafico_lrs(x_opt, f_opt):
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

# Ajustar o Local Random Search (LRS)
def local_random_search_with_plot(n_iterations, bounds, std_dev, patience):
    best = np.random.uniform(low=bounds[0], high=bounds[1], size=2)
    best_eval = objective_function(best)
    no_improvement_count = 0

    for i in range(n_iterations):
        candidate = best + np.random.normal(0, std_dev, size=2)
        candidate = np.clip(candidate, bounds[0], bounds[1])
        candidate_eval = objective_function(candidate)

        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            break

    return best, best_eval

# Parâmetros
R = 30  # Número de repetições
std_dev = 0.1
patience = 100
n_iterations = 10000
bounds = [-100, 100]
melhor_solucao_lrs = None
melhor_valor_lrs = float('inf')
resultados_lrs = []

# Executar o algoritmo Local Random Search (LRS)
for r in range(R):
    print(f"\nExecução {r + 1}/{R}:")
    x_result_lrs, f_result_lrs = local_random_search_with_plot(n_iterations, bounds, std_dev, patience)

    resultados_lrs.append(f_result_lrs)

    if f_result_lrs < melhor_valor_lrs:
        melhor_valor_lrs = f_result_lrs
        melhor_solucao_lrs = x_result_lrs

# Exibir a melhor solução e o resultado mais frequente
print(f"\nMelhor solução encontrada: x1 = {melhor_solucao_lrs[0]:.3f}, x2 = {melhor_solucao_lrs[1]:.3f}")
print(f"Valor mínimo da função: {melhor_valor_lrs:.6f}")

# Mostrar o gráfico da melhor solução encontrada
mostrar_grafico_lrs(melhor_solucao_lrs, melhor_valor_lrs)

resultado_mais_frequente_lrs = Counter(resultados_lrs).most_common(1)[0]
print(f"O resultado mais frequente foi {resultado_mais_frequente_lrs[0]:.6f}, encontrado {resultado_mais_frequente_lrs[1]} vezes.")
