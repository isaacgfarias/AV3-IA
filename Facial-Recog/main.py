import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ajustar para o caminho dos dados
data_dir = r"C:\Users\isaac\Local-Docs\Faculdade\AV3-Cirilo\Facial-Recog\assets"

# Função para carregar as imagens
def load_images(data_dir, img_size):
    X, Y = [], []
    class_labels = sorted(os.listdir(data_dir))
    label_map = {label: idx for idx, label in enumerate(class_labels)}

    for label in class_labels:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (img_size, img_size))
            X.append(img_resized.flatten())
            Y.append(label_map[label])
    
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, label_map

# Função para one-hot encoding
def one_hot_encode(labels, num_classes):
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded

# Funções de RNA: Perceptron Simples, ADALINE, e MLP
class Perceptron:
    def __init__(self, input_size, output_size, learning_rate=0.01, max_epochs=100):
        self.weights = np.random.randn(input_size, output_size)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, Y):
        for epoch in range(self.max_epochs):
            outputs = self.predict(X)
            error = Y - outputs
            self.weights += self.learning_rate * X.T @ error

    def predict(self, X):
        net_input = X @ self.weights
        return np.where(net_input > 0, 1, 0)

class ADALINE:
    def __init__(self, input_size, output_size, learning_rate=0.01, max_epochs=100):
        self.weights = np.random.randn(input_size, output_size)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, Y):
        for epoch in range(self.max_epochs):
            outputs = X @ self.weights
            error = Y - outputs
            self.weights += self.learning_rate * X.T @ error

    def predict(self, X):
        return np.where(X @ self.weights > 0, 1, 0)

# MLP seria implementado de forma semelhante

# Simulações de Monte Carlo
def monte_carlo_simulation(X, Y, model, num_rounds=50):
    metrics = {"accuracy": [], "sensitivity": [], "specificity": []}
    for _ in range(num_rounds):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_idx, test_idx = indices[:split], indices[split:]

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        model.train(X_train, Y_train)
        predictions = model.predict(X_test)

        # Cálculo das métricas
        tp = np.sum((predictions == 1) & (Y_test == 1))
        tn = np.sum((predictions == 0) & (Y_test == 0))
        fp = np.sum((predictions == 1) & (Y_test == 0))
        fn = np.sum((predictions == 0) & (Y_test == 1))

        accuracy = (tp + tn) / len(Y_test)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics["accuracy"].append(accuracy)
        metrics["sensitivity"].append(sensitivity)
        metrics["specificity"].append(specificity)

    return metrics

# Gráficos: matriz de confusão e curva de aprendizado
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Main script
img_sizes = [50, 40, 30, 20, 10]
best_size = None
best_accuracy = 0

for size in img_sizes:
    X, Y, label_map = load_images(data_dir, size)
    X = np.hstack([X, np.ones((X.shape[0], 1))])  # Adiciona o bias
    Y_encoded = one_hot_encode(Y, len(label_map))
    
    perceptron = Perceptron(input_size=X.shape[1], output_size=Y_encoded.shape[1])
    metrics = monte_carlo_simulation(X, Y_encoded, perceptron)

    avg_accuracy = np.mean(metrics["accuracy"])
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_size = size

print(f"Melhor tamanho de imagem: {best_size}")
