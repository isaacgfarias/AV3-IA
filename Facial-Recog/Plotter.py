import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """Classe para gerenciar a visualização dos dados e do modelo."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.x_axis = np.linspace(-2, 10)

    def plot_data(self):
        """Plota os dados iniciais."""
        X, Y = self.dataset.X.T, self.dataset.Y.T
        plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1],
                    s=90, marker='*', color='blue', label='Classe +1')
        plt.scatter(X[Y[:, 0] == -1, 0], X[Y[:, 0] == -1, 1],
                    s=90, marker='s', color='red', label='Classe -1')
        plt.legend()
        plt.ylim(-0.5, 7)
        plt.xlim(-0.5, 7)

    def plot_decision_boundary(self, weights, color='orange', alpha=0.1):
        """Plota a fronteira de decisão do modelo."""
        x2 = -weights[1, 0] / weights[2, 0] * self.x_axis + weights[0, 0] / weights[2, 0]
        x2 = np.nan_to_num(x2)
        plt.plot(self.x_axis, x2, color=color, alpha=alpha)
        plt.pause(0.01)

    def show_final_model(self, weights):
        """Mostra o modelo final após o treinamento."""
        x2 = -weights[1, 0] / weights[2, 0] * self.x_axis + weights[0, 0] / weights[2, 0]
        x2 = np.nan_to_num(x2)
        plt.plot(self.x_axis, x2, color='green', linewidth=3)
        plt.show()