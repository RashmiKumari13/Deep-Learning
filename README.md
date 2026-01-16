Deep Learning Fundamentals: From Perceptrons to Backpropagation

This repository documents my journey through Deep Learning,
starting from first principles. It contains manual
implementations of neural network components using only
NumPy and Matplotlib.

🚀 Experiments Overview1.

   1.Single-Layer Perceptron (SLP)

       Goal: Implement a basic Perceptron for binary classification.
       Key Learning: Understood the Perceptron Update Rule: w = w + n(y -y^)x.
       
       Observation: Successfully classified linearly separable data but failed on non-linear
                   datasets (like XOR), proving the need for multi-layer architectures.
  
  2. Multi-Layer Perceptron (MLP) - Forward Pass

           Goal: Build a multi-layer structure with hidden layers.
           Key Learning: Implemented matrix transformations (Z = WX + b) and non-linear activation functions (Sigmoid).
           Observation: Learned how data flows through a network to create complex representations.

  4. Manual Backpropagation

          Goal: Implement the training algorithm from scratch without using frameworks like PyTorch or TensorFlow.
          Key Learning: Applied the Chain Rule to calculate gradients and update weights across multiple layers.
          Result: The model successfully learned non-linear boundaries (circular/XOR patterns) that the SLP could not solve.
       
       
 🛠️ Tech Stack
      Python 3.x
      NumPy (Linear Algebra)
      Matplotlib (Data Visualization)






