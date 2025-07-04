import numpy as np

class NeuralNetwork:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        return self.sigmoid(np.dot(x, self.weights) + self.bias)
    
    def train(self, X, y, epochs=1000, lr=0.1):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                output = self.forward(xi)
                error = target - output
                self.weights += lr * error * output * (1 - output) * xi
                self.bias += lr * error * output * (1 - output)
    
    def predict(self, X, threshold=0.5):
        return [1 if self.forward(x) >= threshold else 0 for x in X]

# Example usage for AND and OR gates
if __name__ == "__main__":
    # AND gate
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    nn_and = NeuralNetwork(input_size=2)
    nn_and.train(X_and, y_and)
    print("AND Gate Predictions:", nn_and.predict(X_and))
    
    # OR gate
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    
    nn_or = NeuralNetwork(input_size=2)
    nn_or.train(X_or, y_or)
    print("OR Gate Predictions:", nn_or.predict(X_or))