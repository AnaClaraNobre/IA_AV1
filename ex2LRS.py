import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Função a ser maximizada
def f(x):
    return np.exp(-(x[0]**2 + x[1]**2)) + 2 * np.exp(-((x[0] - 1.7)**2 + (x[1] - 1.7)**2))

# Função para gerar candidatos aleatórios no domínio [-2, 4] e [-2, 5]
def gerar_candidato(bounds):
    x1_new = np.random.uniform(bounds[0][0], bounds[0][1])
    x2_new = np.random.uniform(bounds[1][0], bounds[1][1])
    return [x1_new, x2_new]

# Função para mostrar o gráfico ao encontrar um novo valor máximo
def mostrar_grafico(x_opt, f_opt):
    x1_range = np.linspace(-2, 4, 400)
    x2_range = np.linspace(-2, 5, 400)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_grid = f([x1_grid, x2_grid])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, z_grid, cmap='viridis', alpha=0.6)
    ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100)  
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    plt.title(f'Novo valor máximo: f(x1, x2) = {f_opt:.6f}')
    plt.show()

# Função para executar o Local Random Search (LRS) para maximização
def local_random_search(bounds, n_iterations, patience):
    # Gerar solução inicial aleatória
    x_opt = gerar_candidato(bounds)
    f_opt = f(x_opt)
    
    sem_melhoria = 0  # Contador para critério de parada
    for i in range(n_iterations):
        # Gerar um novo candidato
        x_cand = gerar_candidato(bounds)
        f_cand = f(x_cand)
        
        # Se o novo candidato for melhor, atualizamos a solução
        if f_cand > f_opt:
            x_opt = x_cand
            f_opt = f_cand
            sem_melhoria = 0  # Reseta a contagem de sem melhoria
        else:
            sem_melhoria += 1
        
        # Parada antecipada se não houver melhoria por "patience" iterações
        if sem_melhoria >= patience:
            break

    return [round(x, 3) for x in x_opt], round(f_opt, 3)

# parâmetros
bounds = [(-2, 4), (-2, 5)] 
n_iterations = 10000  
patience = 100  
R = 100  


resultados = []
melhor_solucao = None
melhor_valor = float('-inf')

tempo_total_inicio = time.time()

# Executar o algoritmo LRS R vezes e mostrar o gráfico quando um novo valor máximo for encontrado
for r in range(R):
    print(f"\nExecução {r + 1}/{R}:")
    x_result, f_result = local_random_search(bounds, n_iterations, patience)
    
    resultados.append(f_result)
    print(f"Rodada {r + 1}: Solução = {x_result}, f(x) = {f_result}")

    # Se o resultado atual for melhor que o melhor valor global até agora, exibir o gráfico
    if f_result > melhor_valor:
        melhor_valor = f_result
        melhor_solucao = x_result
        # Mostrar o gráfico quando uma nova solução melhor for encontrada

tempo_total_fim = time.time()
tempo_total = tempo_total_fim - tempo_total_inicio
# Exibir a melhor solução encontrada após todas as execuções
print(f"\nMelhor solução encontrada: x1 = {melhor_solucao[0]}, x2 = {melhor_solucao[1]}")
print(f"Valor máximo da função: {melhor_valor}")
mostrar_grafico(melhor_solucao, melhor_valor)

resultado_mais_frequente = Counter(resultados).most_common(1)[0]

# Exibir o resultado mais frequente
print(f"O resultado mais frequente foi {resultado_mais_frequente[0]} encontrado {resultado_mais_frequente[1]} vezes.")
print(f"\nTempo total de execução: {tempo_total:.4f} segundos")
