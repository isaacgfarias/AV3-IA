import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
data = np.loadtxt(r'assets\spiral.csv', delimiter=',')

X = np.array(data[:, :2].T)
Y = np.array(data[:, 2].T)

plt.scatter(X[0], X[1], color='darkorange')

p, N = X.shape

# Normalização dos dados
X_min = X.min(axis=1, keepdims=True)
X_max = X.max(axis=1, keepdims=True)
X = (X - X_min) / (X_max - X_min)

# Adicionar bias
X = np.concatenate((-np.ones((1, N)), X))

Y = Y.astype(int)

# Função para calcular o sinal
def sinal(u_t):
    return np.where(u_t >= 0, 1, -1)

rodadas = 500

# Listas para armazenar as métricas
acuracias = []
sensibilidade = []
especificidade = []
matrizes_de_confusao = []

melhor_treino_erros = []
melhor_teste_erros = []

for i in range(rodadas):
    print(f"Calculando: {i+1}/{rodadas}", end='\r')

    # Inicialização dos pesos com valores pequenos
    W = np.random.uniform(-0.01, 0.01, (p+1, 1))

    # Separação do conjunto de dados em treino e teste
    seed = np.random.permutation(N)
    Xr = X[:, seed]
    yr = Y[seed]

    treino_size = int(N * 0.8)
    X_treino, y_treino = Xr[:, :treino_size], yr[:treino_size]
    X_teste, y_teste = Xr[:, treino_size:], yr[treino_size:]

    # Treinamento do modelo
    aprendizado = 0.01
    treino_erros = []
    teste_erros = []

    for _ in range(1000):  # Limite de 1000 épocas
        # Forward pass e cálculo do erro no conjunto de treino
        u_treino = W.T @ X_treino
        y_pred_treino = sinal(u_treino).flatten()
        erros = y_treino - y_pred_treino
        treino_erros.append(np.mean(np.abs(erros)))  # Erro médio absoluto no treino

        # Cálculo do erro no conjunto de teste
        u_teste = W.T @ X_teste
        y_pred_teste = sinal(u_teste).flatten()
        teste_erros.append(np.mean(np.abs(y_teste - y_pred_teste)))  # Erro médio absoluto no teste

        # Atualizar os pesos
        if not np.any(erros):  # Se não houver erro, pare o treinamento
            break
        W += aprendizado * (X_treino @ erros.reshape(-1, 1)) / treino_size

    # Teste do modelo
    Y_predito = sinal(u_teste).flatten()

    # Métricas
    acertos = np.sum(Y_predito == y_teste)
    acuracias.append(acertos / len(y_teste))

    matriz_confusao = np.array([[0, 0], [0, 0]])
    for j in range(len(y_teste)):
        if Y_predito[j] == 1 and y_teste[j] == 1:
            matriz_confusao[0, 0] += 1  # Verdadeiro positivo (TP)
        elif Y_predito[j] == -1 and y_teste[j] == -1:
            matriz_confusao[1, 1] += 1  # Verdadeiro negativo (TN)
        elif Y_predito[j] == 1 and y_teste[j] == -1:
            matriz_confusao[0, 1] += 1  # Falso positivo (FP)
        elif Y_predito[j] == -1 and y_teste[j] == 1:
            matriz_confusao[1, 0] += 1  # Falso negativo (FN)

    # Sensibilidade e especificidade
    TP = matriz_confusao[0, 0]
    TN = matriz_confusao[1, 1]
    FP = matriz_confusao[0, 1]
    FN = matriz_confusao[1, 0]

    sensibilidade.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
    especificidade.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
    matrizes_de_confusao.append(matriz_confusao)

    # Armazenar erros da melhor rodada (rodada de maior acurácia)
    if i == np.argmax(acuracias):
        melhor_treino_erros = treino_erros
        melhor_teste_erros = teste_erros

# Encontrar as rodadas de maior e menor acurácia
rodada_max_acuracia = np.argmax(acuracias)
rodada_min_acuracia = np.argmin(acuracias)	

sns.heatmap(matrizes_de_confusao[rodada_max_acuracia], annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Real: 1', 'Real: -1'], yticklabels=['Previsão: 1', 'Previsão: -1'])
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Maior Acurácia')
plt.show()

sns.heatmap(matrizes_de_confusao[rodada_min_acuracia], annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real: 1', 'Real: -1'], yticklabels=['Previsão: 1', 'Previsão: -1'])
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Menor Acurácia')
plt.show()

# Exibir a curva de aprendizado da melhor rodada
plt.plot(melhor_treino_erros, label="Erro de Treino")
plt.plot(melhor_teste_erros, label="Erro de Teste")
plt.title("Curva de Aprendizado - Melhor Rodada")
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.legend()
plt.show()

plt.tight_layout()

print(
f'''Acurácia
    Média: {np.mean(acuracias):.4f} 
    DP: {np.std(acuracias):.4f}
    Maior valor: {np.max(acuracias):.4f}
    Menor valor: {np.min(acuracias):.4f}''', end='\n\n')

print(
f'''Sensibilidade
    Média: {np.mean(sensibilidade):.4f}
    DP: {np.std(sensibilidade):.4f}
    Maior valor: {np.max(sensibilidade):.4f}
    Menor valor: {np.min(sensibilidade):.4f}''', end='\n\n')

print(
f'''Especificidade
    Média: {np.mean(especificidade):.4f}
    DP: {np.std(especificidade):.4f}
    Maior valor: {np.max(especificidade):.4f}
    Menor valor: {np.min(especificidade):.4f}''', end='\n\n')