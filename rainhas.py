import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time

def contar_conflitos(board):
    conflitos = 0
    n = len(board)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Verifica se as rainhas estão na mesma linha ou na mesma diagonal
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                conflitos += 1
    return conflitos

# numero 1
# Função que retorna f(x) = 28 - h(x), onde h(x) é o número de conflitos
def calcular_fx(board):
    return 28 - contar_conflitos(board)

# Função de perturbação parsimoniosa
def perturbar_parsimonioso(board):
    new_board = board[:]
    n = len(board)
    
    # Escolhe uma rainha aleatória para mover
    i = random.randint(0, n - 1)
    
    # Muda a rainha para uma linha vizinha (mantendo a perturbação controlada)
    nova_linha = new_board[i] + random.choice([-1, 1])  # Move para uma linha próxima (acima ou abaixo)
    
    # para que a rainha não saia dos limites do tabuleiro
    if nova_linha < 0:
        nova_linha = 0
    elif nova_linha >= n:
        nova_linha = n - 1
    
    new_board[i] = nova_linha
    return new_board

# Função de Têmpera Simulada com perturbação parsimoniosa
def tempera_simulada_parsimoniosa(board_inicial, temperatura_inicial, resfriamento, temperatura_final, max_iteracoes, tipo_escalonamento='geometrico', contador_estados=1):
    temperatura = temperatura_inicial
    estado_atual = board_inicial
    valor_atual = contar_conflitos(estado_atual)
    iteracao = 0
   #inclui o item 4
    while temperatura > temperatura_final and iteracao < max_iteracoes:
        novo_estado = perturbar_parsimonioso(estado_atual)
        novo_valor = contar_conflitos(novo_estado)

        # Se o novo estado for melhor ou aceito pela temperatura, atualiza o estado
        if novo_valor < valor_atual or random.uniform(0, 1) < math.exp((valor_atual - novo_valor) / temperatura):
            if novo_estado != estado_atual:
                estado_atual, valor_atual = novo_estado, novo_valor
                print(f"Estado {contador_estados}: {estado_atual}, f(x): {calcular_fx(estado_atual)}")
                contador_estados += 1
        # item 3
        # tipos de escalonamento
        if tipo_escalonamento == 'geometrico':
            temperatura *= resfriamento
        elif tipo_escalonamento == 'linear':
            temperatura -= 0.01  
        elif tipo_escalonamento == 'logaritmico':
            temperatura = temperatura_inicial / math.log(iteracao + 2)  

        iteracao += 1

        if valor_atual == 0:
            break

    return estado_atual

# desenhar o tabuleiro(grafico)
def desenhar_tabuleiro(board):
    n = len(board)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks(np.arange(0.5, n, 1))
    ax.set_yticks(np.arange(0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)

    for i in range(n):
        for j in range(n):
            color = 'white' if (i + j) % 2 == 0 else 'gray'
            ax.add_patch(plt.Rectangle((i, j), 1, 1, color=color))

    for col, row in enumerate(board):
        ax.text(col + 0.5, row + 0.5, '♕', ha='center', va='center', fontsize=24, color='black')

    plt.grid(True)
    plt.show()

# Parâmetros da Têmpera Simulada
# aqui é item 2
temperatura_inicial = 1000
resfriamento = 0.99
temperatura_final = 0.01
max_iteracoes = 1000
max_execucoes_gerais = 10000  

solucoes = []
max_solucoes = 92
total_iteracoes = 0  
contador_estados = 1  

tempos_de_execucao = []

start_time = time.time()

tipo_escalonamento = 'geometrico'  #'linear', 'logaritmico'

# Rodar o algoritmo até encontrar as 92 soluções ou atingir o máximo de execuções
while len(solucoes) < max_solucoes and total_iteracoes < max_execucoes_gerais:
    # Gerar um estado inicial completamente aleatório
    estado_inicial = [random.randint(0, 7) for _ in range(8)]
    
    # Executa a Têmpera Simulada com perturbação parsimoniosa e tipo de escalonamento selecionado
    solucao = tempera_simulada_parsimoniosa(estado_inicial, temperatura_inicial, resfriamento, temperatura_final, max_iteracoes, tipo_escalonamento, contador_estados)
    
    # Se a solução não tiver conflitos, adiciona à lista de soluções
    if contar_conflitos(solucao) == 0:
        solucoes.append(solucao)
        print(f"Solução {len(solucoes)}: {solucao}, f(x): {calcular_fx(solucao)}")
        
        tempos_de_execucao.append(time.time() - start_time)
        
        # mostra o desenho para cada vez que encontra uma solução "perfeita"
        # desenhar_tabuleiro(solucao)
    
    total_iteracoes += 1

# numéro 6
# Medir o tempo final e calcular o tempo total de execução
end_time = time.time()
tempo_total = end_time - start_time
print(f"Total de soluções encontradas: {len(solucoes)}")
print(f"Tempo total de execução: {tempo_total} segundos")

# gráfico de tempo do total da execução 
plt.plot(range(1, len(tempos_de_execucao) + 1), tempos_de_execucao)
plt.title(f'Tempo de Execução para {tipo_escalonamento.capitalize()}')
plt.xlabel('Número de Soluções Encontradas')
plt.ylabel('Tempo de Execução (segundos)')
plt.grid(True)
plt.show()
