import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.costs = []

    def activation(self, X):
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_output = self.activation(X)
            errors = y - linear_output
            cost = (errors ** 2).sum() / (2.0 * n_samples)
            self.costs.append(cost)

            self.weights += self.learning_rate * X.T.dot(errors) / n_samples
            self.bias += self.learning_rate * errors.sum() / n_samples

    def predict(self, X):
        output = self.activation(X)
        return np.where(output >= 0.0, 1, 0)


def load_images(data_dir, img_size):
    X, y = [], []
    label_map = {}
    label_counter = 0

    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            if person not in label_map:
                label_map[person] = label_counter
                label_counter += 1

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img.flatten())
                y.append(label_map[person])

    return np.array(X), np.array(y), label_map


def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    return one_hot


def split_data(X, y, train_ratio=0.8):
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    train_size = int(train_ratio * len(y))
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def monte_carlo_simulation(X, y, model, label_map, num_simulations=50):
    accs, sens, specs = [], [], []
    for _ in range(num_simulations):
        X_train, X_test, y_train, y_test = split_data(X, y)

        num_classes = len(label_map)
        y_train_one_hot = one_hot_encode(y_train, num_classes)

        # Train one Adaline per class (One-vs-All strategy)
        classifiers = [Adaline(learning_rate=0.01, epochs=50) for _ in range(num_classes)]
        for class_label, clf in enumerate(classifiers):
            clf.train(X_train, y_train_one_hot[:, class_label])

        # Testing
        y_pred = np.zeros((len(X_test), num_classes))
        for class_label, clf in enumerate(classifiers):
            y_pred[:, class_label] = clf.predict(X_test)

        y_pred_labels = np.argmax(y_pred, axis=1)

        # Metrics calculation
        acc = np.mean(y_pred_labels == y_test)
        accs.append(acc)

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(y_test, y_pred_labels):
            confusion_matrix[true_label, pred_label] += 1

        sensitivity = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
        specificity = (confusion_matrix.sum() - confusion_matrix.sum(axis=0) - confusion_matrix.sum(axis=1) + np.diag(confusion_matrix)) / (confusion_matrix.sum() - confusion_matrix.sum(axis=1))

        sens.append(np.mean(sensitivity))
        specs.append(np.mean(specificity))

    return accs, sens, specs


def plot_metrics(accs, sens, specs):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(accs, ax=ax[0])
    ax[0].set_title('Accuracy Distribution')
    ax[0].set_xlabel('Accuracy')

    sns.boxplot(sens, ax=ax[1])
    ax[1].set_title('Sensitivity Distribution')
    ax[1].set_xlabel('Sensitivity')

    sns.boxplot(specs, ax=ax[2])
    ax[2].set_title('Specificity Distribution')
    ax[2].set_xlabel('Specificity')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(confusion_matrix, label_map, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


# Main script
if __name__ == "__main__":
    data_dir = "Facial-Recog\assets"  # Replace with your dataset directory
    img_size = 30  # You can experiment with other sizes (50, 40, 20, 10)

    # Load dataset
    X, y, label_map = load_images(data_dir, img_size)

    # Normalize features
    X = X / 255.0

    # Monte Carlo simulations
    model = Adaline(learning_rate=0.01, epochs=50)
    accs, sens, specs = monte_carlo_simulation(X, y, model, label_map)

    # Plot performance metrics
    plot_metrics(accs, sens, specs)

    # Calculate and plot confusion matrix for best and worst cases
    best_case = np.argmax(accs)
    worst_case = np.argmin(accs)

    print(f"Best Accuracy: {accs[best_case]:.2f}, Worst Accuracy: {accs[worst_case]:.2f}")
