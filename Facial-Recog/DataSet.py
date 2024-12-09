import numpy as np

class DataSet:
    """Classe para gerenciar o conjunto de dados."""
    def __init__(self, features, labels):
        self.X = features.T  # Transpor para dimensão (p x N)
        self.Y = labels.T
        self.add_bias()

    def add_bias(self):
        """Adiciona o termo de viés aos dados."""
        self.X = np.concatenate((-np.ones((1, self.X.shape[1])), self.X))