import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
data = np.loadtxt(r'assets\spiral.csv', delimiter=',')
X = np.array(data[:, :2].T)
Y = np.array(data[:, 2].T).astype(int)

# Preparação dos dados
p, N = X.shape
X = np.concatenate((-np.ones((1, N)), X))  # Bias
Y[Y == 0] = -1  # Ajustando as classes para [-1, 1]

# Funções auxiliares
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def sinal(u_t):
    return np.where(u_t >= 0, 1, -1)

# Inicialização Xavier
def inicializa_pesos(fan_in, fan_out):
    limite = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limite, limite, (fan_out, fan_in))

# Parâmetros do MLP
n_hidden = 50  # Número de neurônios em cada camada oculta
rodadas = 2  
aprendizado = 0.001  # Taxa de aprendizado aumentada
epocas = 2000  # Número de épocas aumentado

# Métricas
acuracias = []
sensibilidades = []
especificidades = []
matrizes_de_confusao = []

# Armazenar erros de treino e teste para a melhor rodada
melhor_treino_erros = []
melhor_teste_erros = []

# Rodadas de Monte Carlo
for i in range(rodadas):
    print(f"Rodada {i + 1}/{rodadas}", end='\r')

    # Inicialização dos pesos com Xavier
    W1 = inicializa_pesos(p + 1, n_hidden)  # Pesos da entrada para a primeira camada oculta
    W2 = inicializa_pesos(n_hidden + 1, n_hidden)  # Pesos da primeira para a segunda camada oculta
    W3 = inicializa_pesos(n_hidden + 1, 1)  # Pesos da segunda camada oculta para a saída

    # Separação em treino e teste
    seed = np.random.permutation(N)
    Xr, yr = X[:, seed], Y[seed]
    treino_size = int(N * 0.8)
    X_treino, y_treino = Xr[:, :treino_size], yr[:treino_size]
    X_teste, y_teste = Xr[:, treino_size:], yr[treino_size:]

    treino_erros = []
    teste_erros = []

    # Treinamento do MLP
    for epoch in range(epocas):
        # Forward pass
        Z1 = W1 @ X_treino
        A1 = np.concatenate((-np.ones((1, treino_size)), sigmoid(Z1)))  # Primeira camada oculta com bias
        Z2 = W2 @ A1
        A2 = np.concatenate((-np.ones((1, treino_size)), sigmoid(Z2)))  # Segunda camada oculta com bias
        Z3 = W3 @ A2
        A3 = sigmoid(Z3)  # Saída

        # Backpropagation
        erro_saida = y_treino - A3.flatten()
        grad_W3 = (erro_saida * sigmoid_derivative(Z3)) @ A2.T
        erro_oculto_2 = (W3[:, 1:].T @ (erro_saida * sigmoid_derivative(Z3))) * sigmoid_derivative(Z2)
        grad_W2 = erro_oculto_2 @ A1.T
        erro_oculto_1 = (W2[:, 1:].T @ erro_oculto_2) * sigmoid_derivative(Z1)
        grad_W1 = erro_oculto_1 @ X_treino.T

        # Atualização dos pesos
        W3 += aprendizado * grad_W3
        W2 += aprendizado * grad_W2
        W1 += aprendizado * grad_W1

        # Armazenar erros para a curva de aprendizado
        treino_erros.append(np.mean(np.abs(erro_saida)))

        # Calcular erro no conjunto de teste
        Z1_test = W1 @ X_teste
        A1_test = np.concatenate((-np.ones((1, X_teste.shape[1])), sigmoid(Z1_test)))
        Z2_test = W2 @ A1_test
        A2_test = np.concatenate((-np.ones((1, X_teste.shape[1])), sigmoid(Z2_test)))
        Z3_test = W3 @ A2_test
        Y_predito_test = sigmoid(Z3_test)
        teste_erros.append(np.mean(np.abs(y_teste - Y_predito_test.flatten())))

    # Teste do MLP
    Z1 = W1 @ X_teste
    A1 = np.concatenate((-np.ones((1, X_teste.shape[1])), sigmoid(Z1)))
    Z2 = W2 @ A1
    A2 = np.concatenate((-np.ones((1, X_teste.shape[1])), sigmoid(Z2)))
    Z3 = W3 @ A2
    Y_predito = sinal(Z3).flatten()

    # Métricas
    acertos = np.sum(Y_predito == y_teste)
    acuracias.append(acertos / len(y_teste))

    matriz_confusao = np.array([[0, 0], [0, 0]])
    for j in range(len(y_teste)):
        if Y_predito[j] == 1 and y_teste[j] == 1:
            matriz_confusao[0, 0] += 1
        elif Y_predito[j] == -1 and y_teste[j] == -1:
            matriz_confusao[1, 1] += 1
        elif Y_predito[j] == 1 and y_teste[j] == -1:
            matriz_confusao[0, 1] += 1
        elif Y_predito[j] == -1 and y_teste[j] == 1:
            matriz_confusao[1, 0] += 1

    TP = matriz_confusao[0, 0]
    TN = matriz_confusao[1, 1]
    FP = matriz_confusao[0, 1]
    FN = matriz_confusao[1, 0]

    sensibilidades.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
    especificidades.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
    matrizes_de_confusao.append(matriz_confusao)

    # Armazenar erros da melhor rodada
    if i == np.argmax(acuracias):
        melhor_treino_erros = treino_erros
        melhor_teste_erros = teste_erros

# Resultados
rodada_max_acuracia = np.argmax(acuracias)
rodada_min_acuracia = np.argmin(acuracias)

sns.heatmap(matrizes_de_confusao[rodada_max_acuracia], annot=True, fmt='d', cmap='Greens',
            xticklabels=['Real: 1', 'Real: -1'], yticklabels=['Previsão: 1', 'Previsão: -1'])
plt.title('Matriz de Confusão - Maior Acurácia')
plt.show()

sns.heatmap(matrizes_de_confusao[rodada_min_acuracia], annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real: 1', 'Real: -1'], yticklabels=['Previsão: 1', 'Previsão: -1'])
plt.title('Matriz de Confusão - Menor Acurácia')
plt.show()

# Curva de aprendizado da melhor rodada
plt.plot(melhor_treino_erros, label="Erro de Treino")
plt.plot(melhor_teste_erros, label="Erro de Teste")
plt.title("Curva de Aprendizado - Melhor Rodada")
plt.xlabel("Épocas")
plt.ylabel("Erro")
plt.legend()
plt.show()

# Estatísticas
# print(f"Acurácia: Média={np.mean(acuracias):.4f}, DP={np.std(acuracias):.4f}, Max={np.max(acuracias):.4f}, Min={np.min(acuracias):.4f}")
# print(f"Sensibilidade: Média={np.mean(sensibilidades):.4f}, DP={np.std(sensibilidades):.4f}, Max={np.max(sensibilidades):.4f}, Min={np.min(sensibilidades):.4f}")
# print(f"Especificidade: Média={np.mean(especificidades):.4f}, DP={np.std(especificidades):.4f}, Max={np.max(especificidades):.4f}, Min={np.min(especificidades):.4f}")

print(
f'''Acurácia
    Média: {np.mean(acuracias):.4f} 
    DP: {np.std(acuracias):.4f}
    Maior valor: {np.max(acuracias):.4f}
    Menor valor: {np.min(acuracias):.4f}''', end='\n\n')

print(
f'''Sensibilidade
    Média: {np.mean(sensibilidades):.4f}
    DP: {np.std(sensibilidades):.4f}
    Maior valor: {np.max(sensibilidades):.4f}
    Menor valor: {np.min(sensibilidades):.4f}''', end='\n\n')

print(
f'''Especificidade
    Média: {np.mean(especificidades):.4f}
    DP: {np.std(especificidades):.4f}
    Maior valor: {np.max(especificidades):.4f}
    Menor valor: {np.min(especificidades):.4f}''', end='\n\n')



