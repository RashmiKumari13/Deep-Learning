#w=w-n(y-y^)x
#w=[w0 w1 w2]
"""x=[1 x1 x2 ,  1 ...  ... , 1   ... ...][]
    
    y^=w0x0+w1x1+w2x2
    y^=wx^t

    [w0 w1 w2][ 1
                x1
                x2]

"""
import numpy as np
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
    def fit(self, X, y):       
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            for i in range(len(X)):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(linear_output)
                self.weights += self.lr * (y[i] - y_pred) * X[i]
                self.bias += self.lr * (y[i] - y_pred)
    def activation(self, x):
        return 1 if x >= 0 else 0
    def predict(self, X):
        predictions = []
        for x in X:
            linear_output = np.dot(x, self.weights) + self.bias
            predictions.append(self.activation(linear_output))
        return predictions
X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
)
y = np.array([0, 0, 0, 1])
p = Perceptron(learning_rate=0.1, epochs=20)
p.fit(X, y)
print("Predictions:")
print(p.predict(X))


##
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X = np.random.randn(1000, 2)
 
y = np.where(2*X[:, 0] + 3*X[:, 1] - 1 > 0, 1, 0)

class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        
        for epoch in range(self.epochs):
            for i in range(len(X)):                
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                
                update = self.lr * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)
model = Perceptron(lr=0.1, epochs=20)
model.fit(X, y)

predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Final Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(10, 6))

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', alpha=0.5)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', alpha=0.5)

x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(model.weights[0] * x_vals + model.bias) / model.weights[1]
plt.plot(x_vals, y_vals, color='black', linewidth=2, label='Decision Boundary')

plt.title("SLP Binary Classification (1000 Data Points)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()