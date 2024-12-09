
import numpy as np

# Modelo Perceptron Simples
class Perceptron:
    """Classe para implementar e treinar o Perceptron Simples."""
    def __init__(self, input_dim, learning_rate=0.001):
        self.lr = learning_rate
        self.weights = np.random.random_sample((input_dim + 1, 1)) - 0.5

    def train(self, dataset, plotter):
        """Treina o perceptron e atualiza os pesos."""
        erro = True
        epoca = 0
        p, N = dataset.X.shape

        while erro:
            erro = False
            for t in range(N):
                x_t = dataset.X[:, t].reshape(p + 1, 1)
                u_t = (self.weights.T @ x_t)[0, 0]
                y_t = SignFunction.activate(u_t)
                d_t = float(dataset.Y[0, t])
                e_t = d_t - y_t
                self.weights += (self.lr * e_t * x_t) / 2
                if y_t != d_t:
                    erro = True

            # Atualiza visualização após cada época
            plotter.plot_decision_boundary(self.weights)
            epoca += 1

        return epoca

    def predict(self, x):
        """Realiza predições para novos dados."""
        u = self.weights.T @ x
        return SignFunction.activate(u)



class SignFunction:
    """Classe para a função de ativação (Sign Function)."""
    @staticmethod
    def activate(u):
        return 1 if u >= 0 else -1