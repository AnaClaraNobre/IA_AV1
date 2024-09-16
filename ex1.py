import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Função a ser minimizada
def f(x):
    return x[0]**2 + x[1]**2

# Função de perturbação que gera vizinhos aleatórios em torno da solução atual
def perturb(x, e):
    return [x[0] + np.random.uniform(-e, e), x[1] + np.random.uniform(-e, e)]

# Função para executar o Hill Climbing com gráficos
def hill_climbing(e, max_it, max_viz, t, plot_graphs=False):
    x_opt = [np.random.uniform(-100, 100), np.random.uniform(-100, 100)]
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
            if f_cand < f_opt:  # Critério de minimização
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

    # Mostrar gráficos se necessário
    if plot_graphs:
        # Gráfico 1: Evolução do valor mínimo ao longo das iterações
        plt.figure()
        plt.plot(valores)
        plt.xlabel('Iterações')
        plt.ylabel('Valor da função')
        plt.title('Evolução do valor mínimo')
        plt.show()

        # Gráfico 2: Trajetória da busca no espaço 2D
        plt.figure()
        plt.scatter(x1_vals, x2_vals, c='r', marker='x')
        plt.scatter(x_opt[0], x_opt[1], color='g', marker='x', s=100, linewidth=3)  # Ponto final em verde
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Trajetória da busca no espaço 2D')
        plt.show()

        # Gráfico 3: Superfície 3D da função e ponto final em verde
        x1_range = np.linspace(-100, 100, 400)
        x2_range = np.linspace(-100, 100, 400)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        z_grid = f([x1_grid, x2_grid])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1_grid, x2_grid, z_grid, cmap='viridis', alpha=0.6)
        ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100)  
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1, x2)')
        ax.set_xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
        ax.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
        ax.set_zticks([2500,5000,7500,10000,12500,15000,17500,20000])
        plt.title('Gráfico da função')
        plt.show()

    return [round(x, 3) for x in x_opt], round(f_opt, 3)

#parâmetros
e = 1.0  
max_it = 10000  # Número máximo de iterações
max_viz = 20  
t = 100  
R = 50 

resultados = []
melhor_solucao = None
melhor_valor = float('inf')

# Executar o algoritmo R vezes e mostrar gráficos da última execução
for r in range(R):
    if r == R - 1:  # Mostrar gráficos apenas na última execução
        x_result, f_result = hill_climbing(e, max_it, max_viz, t, plot_graphs=True)
    else:
        x_result, f_result = hill_climbing(e, max_it, max_viz, t)
    
    resultados.append(f_result)

    if f_result < melhor_valor:
        melhor_valor = f_result
        melhor_solucao = x_result

# Exibir a melhor solução encontrada após todas as execuções
print(f"Melhor solução encontrada: x1 = {melhor_solucao[0]}, x2 = {melhor_solucao[1]}")
print(f"Valor mínimo da função: {melhor_valor}")

resultado_mais_frequente = Counter(resultados).most_common(1)[0]

print(f"O resultado mais frequente foi {resultado_mais_frequente[0]} encontrado {resultado_mais_frequente[1]} vezes.")
