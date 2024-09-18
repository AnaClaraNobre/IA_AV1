import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Função a ser maximizada
def f(x):
    return np.exp(-(x[0]**2 + x[1]**2)) + 2 * np.exp(-((x[0] - 1.7)**2 + (x[1] - 1.7)**2))

# Função de perturbação para gerar vizinhos aleatórios no domínio [-2, 4] e [-2, 5]
def perturb(x, e):
    x1_new = np.clip(x[0] + np.random.uniform(-e, e), -2, 4) 
    x2_new = np.clip(x[1] + np.random.uniform(-e, e), -2, 5)  
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
    plt.title(f'Novo valor máximo: f(x1, x2) = {f_opt}')
    plt.show()

# Função para executar o Hill Climbing para maximização
def hill_climbing(e, max_it, max_viz, t):
    x_opt = [np.random.uniform(-2, 4), np.random.uniform(-2, 5)]
    f_opt = f(x_opt)
    
    sem_melhoria = 0  
    valores = [f_opt]
    x1_vals, x2_vals = [], []  

    i = 0
    while i < max_it:
        melhoria = False
        for _ in range(max_viz):
            x_cand = perturb(x_opt, e)
            f_cand = f(x_cand)
            if f_cand > f_opt:  
                x_opt = x_cand
                f_opt = f_cand
                valores.append(f_opt)
                sem_melhoria = 0  
                melhoria = True
                break

        if not melhoria:
            sem_melhoria += 1
        
        if sem_melhoria >= t:
            break
        
        i += 1
        x1_vals.append(x_opt[0])
        x2_vals.append(x_opt[1])

    return [round(x, 3) for x in x_opt], round(f_opt, 3)

# Definir parâmetros
e = 0.1  
max_it = 10000  # Número máximo de iterações
max_viz = 20 
t = 100  
R = 100 

resultados = []
melhor_solucao = None
melhor_valor = float('-inf')

tempo_total_inicio = time.time()
# Executar o algoritmo R vezes e mostrar o gráfico quando um novo valor máximo for encontrado
for r in range(R):
    print(f"\nExecução {r + 1}/{R}:")
    x_result, f_result = hill_climbing(e, max_it, max_viz, t)
    
    resultados.append(f_result)
    print(f"Rodada {r + 1}: Solução = {x_result}, f(x) = {f_result}")

    # Se o resultado atual for melhor que o melhor valor global até agora, exibir o gráfico
    if f_result > melhor_valor:
        melhor_valor = f_result
        melhor_solucao = x_result
        # Mostrar o gráfico quando uma nova solução melhor for encontrada
        mostrar_grafico(melhor_solucao, melhor_valor)

tempo_total_fim = time.time()
tempo_total = tempo_total_fim - tempo_total_inicio

# Exibir a melhor solução encontrada após todas as execuções
print(f"\nMelhor solução encontrada: x1 = {melhor_solucao[0]}, x2 = {melhor_solucao[1]}")
print(f"Valor máximo da função: {melhor_valor}")

resultado_mais_frequente = Counter(resultados).most_common(1)[0]

# Exibir o resultado mais frequente
print(f"O resultado mais frequente foi {resultado_mais_frequente[0]} encontrado {resultado_mais_frequente[1]} vezes.")
print(f"\nTempo total de execução: {tempo_total:.4f} segundos")
