import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Nova função a ser minimizada
def f(x):
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    return term1 + term2 + 20 + np.e

def perturb(x, e):
    x1_new = np.clip(x[0] + np.random.uniform(-e, e), -8, 8)  
    x2_new = np.clip(x[1] + np.random.uniform(-e, e), -8, 8)  
    return [x1_new, x2_new]

# Função para mostrar o gráfico ao encontrar um novo valor mínimo
def mostrar_grafico(x_opt, f_opt):
    x1_range = np.linspace(-8, 8, 400)
    x2_range = np.linspace(-8, 8, 400)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_grid = f([x1_grid, x2_grid])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, z_grid, cmap='viridis', alpha=0.6)
    ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100)  
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    plt.title(f'Novo valor mínimo: f(x1, x2) = {f_opt}')
    plt.show()

# Função para executar o Hill Climbing com gráficos
def hill_climbing(e, max_it, max_viz, t, plot_graphs=False):
    x_opt = [np.random.uniform(-8, 8), np.random.uniform(-8, 8)]  
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
        plt.scatter(x_opt[0], x_opt[1], color='g', marker='x', s=100, linewidth=3)  
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Trajetória da busca no espaço 2D')
        plt.show()

        # Gráfico 3: Superfície 3D da função e ponto final em verde
        x1_range = np.linspace(-8, 8, 400)
        x2_range = np.linspace(-8, 8, 400)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        z_grid = f([x1_grid, x2_grid])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1_grid, x2_grid, z_grid, cmap='viridis', alpha=0.6)
        ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100)  
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1, x2)')
        plt.title('Gráfico da função')
        plt.show()

    return [round(x, 3) for x in x_opt], round(f_opt, 3)

# Definir parâmetros
e = 1.0  
max_it = 10000  # Número máximo de iterações
max_viz = 20  
t = 100  
R = 50 

resultados = []
melhor_solucao = None
melhor_valor = float('inf')

# Executar o algoritmo R vezes e mostrar o gráfico quando um novo valor mínimo for encontrado
for r in range(R):
    print(f"\nExecução {r + 1}/{R}:")
    x_result, f_result = hill_climbing(e, max_it, max_viz, t)
    
    resultados.append(f_result)

    # Se o resultado atual for melhor que o melhor valor global até agora, exibir o gráfico
    if f_result < melhor_valor:
        melhor_valor = f_result
        melhor_solucao = x_result
        # Mostrar o gráfico quando uma nova solução melhor for encontrada
        mostrar_grafico(melhor_solucao, melhor_valor)

# Exibir a melhor solução encontrada após todas as execuções
print(f"\nMelhor solução encontrada: x1 = {melhor_solucao[0]}, x2 = {melhor_solucao[1]}")
print(f"Valor mínimo da função: {melhor_valor}")

resultado_mais_frequente = Counter(resultados).most_common(1)[0]

# Exibir o resultado mais frequente
print(f"O resultado mais frequente foi {resultado_mais_frequente[0]} encontrado {resultado_mais_frequente[1]} vezes.")
