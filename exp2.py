import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.randn(1000, 2)

y = np.where((X[:, 0]**2 + X[:, 1]**2) < 1.5, 1, 0).reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class MLP_Forward:
    def __init__(self, input_size, hidden_size, output_size):
       
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward_pass(self, X):
        
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
       
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

mlp_fwd = MLP_Forward(input_size=2, hidden_size=4, output_size=1)
initial_predictions = mlp_fwd.forward_pass(X)

print("Forward Pass complete.")
print(f"Shape of output: {initial_predictions.shape}")
print(f"Sample prediction (before training): {initial_predictions[0]}")