import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.randn(1000, 2)

y = np.where((X[:, 0]**2 + X[:, 1]**2) < 1.5, 1, 0).reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

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


class MLP_Backprop(MLP_Forward):
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        
        super().__init__(input_size, hidden_size, output_size)
        self.lr = lr

    def backward_pass(self, X, y, output):
       
        error_output = y - output
        delta_output = error_output * sigmoid_derivative(output)

        # 2 (Backpropagation)
        error_hidden = delta_output.dot(self.W2.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.a1)

        # 3. Weight Updates 
        self.W2 += self.a1.T.dot(delta_output) * self.lr
        self.b2 += np.sum(delta_output, axis=0, keepdims=True) * self.lr
        self.W1 += X.T.dot(delta_hidden) * self.lr
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.lr

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward_pass(X)
            self.backward_pass(X, y, output)
            if epoch % 2000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

nn = MLP_Backprop(input_size=2, hidden_size=8, output_size=1, lr=0.01)
print("Starting Training...")
nn.train(X, y, epochs=10000)

final_out = nn.forward_pass(X)
final_preds = np.where(final_out >= 0.5, 1, 0)
accuracy = np.mean(final_preds == y)
print(f"\nFinal Accuracy after Backpropagation: {accuracy * 100:.2f}%")