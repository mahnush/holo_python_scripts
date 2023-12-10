import matplotlib.pyplot as plt
import numpy as np

# Define the function f
def f(x1, x2):
    return x1 + 3*x2

# Define the ranges of x1 and x2 to plot
x1_vals = np.linspace(-10, 10, 100)
x2_vals = np.linspace(-10, 10, 100)

# Create a meshgrid of x1 and x2 values
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Compute the values of f for each combination of x1 and x2
Z = f(X1, X2)

# Create a 3D plot of f(x1, x2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X1, X2, Z)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
plt.show()