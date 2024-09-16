import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Função a ser minimizada (com reescalonamento)
def f(x):
    x1, x2 = x[0], x[1]
    # Reescalonamento dos expoentes para evitar overflow
    term1 = -np.sin(x1) * np.sin((x1**2 / np.pi)**(2**4))  # Reescalonado para evitar overflow
    term2 = -np.sin(x2) * np.sin((2 * x2**2 / np.pi)**(2**4))  # Reescalonado para evitar overflow
    return term1 + term2

# Função para gerar novos candidatos aleatórios dentro do domínio [0, pi]
def gerar_candidato():
    x1_new = np.random.uniform(0, np.pi)
    x2_new = np.random.uniform(0, np.pi)
    return [x1_new, x2_new]

# Função para mostrar o gráfico ao encontrar um novo valor mínimo
def mostrar_grafico(x_opt, f_opt):
    x1_range = np.linspace(0, np.pi, 400)
    x2_range = np.linspace(0, np.pi, 400)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_grid = f([x1_grid, x2_grid])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, z_grid, cmap='viridis', alpha=0.6)
    ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100)  
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    plt.title(f'Novo valor mínimo: f(x1, x2) = {f_opt:.6f}')
    plt.show()

# Função para executar o Local Random Search (LRS) para minimização
def local_random_search(n_iterations, patience):
    x_opt = gerar_candidato()  # Solução inicial aleatória
    f_opt = f(x_opt)
    
    sem_melhoria = 0  
    for i in range(n_iterations):
        # Gerar um novo candidato aleatório
        x_cand = gerar_candidato()
        f_cand = f(x_cand)
        
        # Se o novo candidato for melhor, atualizamos a solução
        if f_cand < f_opt:
            x_opt = x_cand
            f_opt = f_cand
            sem_melhoria = 0  
        else:
            sem_melhoria += 1
        
        # Critério de parada por iterações sem melhoria
        if sem_melhoria >= patience:
            break

    return [round(x, 3) for x in x_opt], round(f_opt, 3)

# Definir parâmetros
n_iterations = 10000  # Número máximo de iterações
patience = 100  # Número de iterações sem melhoria antes de parar
R = 50  # Número de repetições do algoritmo

resultados = []
melhor_solucao = None
menor_valor = float('inf')

# Executar o algoritmo Local Random Search (LRS)
for r in range(R):
    print(f"\nExecução {r + 1}/{R}:")
    x_result, f_result = local_random_search(n_iterations, patience)
    
    resultados.append(f_result)

    # Se o resultado atual for menor que o menor valor global até agora, exibir o gráfico
    if f_result < menor_valor:
        menor_valor = f_result
        melhor_solucao = x_result
        # Mostrar o gráfico quando uma nova solução melhor for encontrada

# Exibir a melhor solução encontrada após todas as execuções
print(f"\nMelhor solução encontrada: x1 = {melhor_solucao[0]}, x2 = {melhor_solucao[1]}")
print(f"Valor mínimo da função: {menor_valor}")
mostrar_grafico(melhor_solucao, menor_valor)

resultado_mais_frequente = Counter(resultados).most_common(1)[0]

# Exibir o resultado mais frequente
print(f"O resultado mais frequente foi {resultado_mais_frequente[0]} encontrado {resultado_mais_frequente[1]} vezes.")
