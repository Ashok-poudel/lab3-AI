import numpy as np

class BackpropagationNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]))
            self.biases.append(np.random.randn(layers[i+1]))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        activations = [x]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activations.append(self.sigmoid(z))
        return activations
    
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                # Forward pass
                activations = self.forward(xi)
                
                # Backward pass
                errors = [None] * len(self.weights)
                deltas = [None] * len(self.weights)
                
                # Output layer error
                errors[-1] = target - activations[-1]
                deltas[-1] = errors[-1] * self.sigmoid_derivative(activations[-1])
                
                # Hidden layers error
                for i in range(len(self.weights)-2, -1, -1):
                    errors[i] = np.dot(deltas[i+1], self.weights[i+1].T)
                    deltas[i] = errors[i] * self.sigmoid_derivative(activations[i+1])
                
                # Update weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] += learning_rate * np.outer(activations[i], deltas[i])
                    self.biases[i] += learning_rate * deltas[i]
    
    def predict(self, X, threshold=0.5):
        predictions = []
        for x in X:
            output = self.forward(x)[-1]
            predictions.append(1 if output >= threshold else 0)
        return predictions

# Example usage for XOR problem
if __name__ == "__main__":
    # XOR gate (requires hidden layer)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    # Network with 2 input neurons, 2 hidden neurons, and 1 output neuron
    bp_net = BackpropagationNetwork([2, 2, 1])
    bp_net.train(X_xor, y_xor, epochs=10000, learning_rate=0.1)
    
    print("XOR Predictions:", bp_net.predict(X_xor))