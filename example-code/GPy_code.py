import GPy
import numpy as np
from src.gp_from_first_principles.load_data import load_data

# Load your data
y, x = load_data()
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Define the number of inducing points and select them randomly from x (or use a more sophisticated method)
Z = np.random.uniform(x.min(), x.max(), (20, 1))  # 20 inducing points

# Define the kernel with specified hyperparameters
kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)

# Create the Sparse GP model using the FITC approximation
model = GPy.models.SparseGPRegression(X=x, Y=y, kernel=kernel, Z=Z)

# Set the noise variance (Gaussian noise) in the likelihood
model.likelihood.variance = 0.02

# Print the initial NLML before optimization
print("Initial NLML:", model.objective_function())

# Optimize the model (you can specify the optimizer and number of iterations)
model.optimize(messages=True, max_iters=1000)

# Print the optimized NLML
print("Optimized NLML:", model.objective_function())

# Optionally, print the optimized model parameters
print("Optimized model parameters:")
print(model)