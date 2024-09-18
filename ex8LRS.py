import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Função a ser minimizada (da imagem)
def objective_function(x):
    x1, x2 = x[0], x[1]
    term1 = -(x2 + 47) * np.sin(np.sqrt(abs(x1 / 2 + (x2 + 47))))
    term2 = -x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))
    return term1 + term2

# Função para mostrar o gráfico ao encontrar um novo valor mínimo
def mostrar_grafico_lrs(x_opt, f_opt):
    x1_range = np.linspace(-200, 20, 400)
    x2_range = np.linspace(-200, 20, 400)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_grid = objective_function([x1_grid, x2_grid])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, z_grid, cmap='viridis', alpha=0.6)
    ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_xticks([-200, -150, -100, -50, 0, 20])
    ax.set_yticks([-200, -150, -100, -50, 0, 20])
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
R = 100
std_dev = 0.1
patience = 100
n_iterations = 10000
bounds = [-200, 20]
melhor_solucao_lrs = None
melhor_valor_lrs = float('inf')
resultados_lrs = []
tempo_total_inicio = time.time()

# Executar o algoritmo Local Random Search (LRS)
for r in range(R):
    print(f"\nExecução {r + 1}/{R}:")
    x_result_lrs, f_result_lrs = local_random_search_with_plot(n_iterations, bounds, std_dev, patience)

    resultados_lrs.append(f_result_lrs)
    
    print(f"Rodada {r + 1}: Solução = {x_result_lrs}, f(x) = {f_result_lrs:.6f}")


    if f_result_lrs < melhor_valor_lrs:
        melhor_valor_lrs = f_result_lrs
        melhor_solucao_lrs = x_result_lrs

tempo_total_fim = time.time()
tempo_total = tempo_total_fim - tempo_total_inicio

# Exibir a melhor solução e o resultado mais frequente
print(f"\nMelhor solução encontrada: x1 = {melhor_solucao_lrs[0]:.6f}, x2 = {melhor_solucao_lrs[1]:.6f}")
print(f"Valor mínimo da função: {melhor_valor_lrs:.6f}")

# Mostrar o gráfico da melhor solução encontrada
mostrar_grafico_lrs(melhor_solucao_lrs, melhor_valor_lrs)

resultado_mais_frequente_lrs = Counter(resultados_lrs).most_common(1)[0]
print(f"O resultado mais frequente foi {resultado_mais_frequente_lrs[0]:.6f}, encontrado {resultado_mais_frequente_lrs[1]} vezes.")
print(f"\nTempo total de execução: {tempo_total:.4f} segundos")
