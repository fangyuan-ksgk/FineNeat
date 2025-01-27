import matplotlib.pyplot as plt
import numpy as np
from fineNeat.sneat_jax.ann import act
from jax import numpy as jnp
from PIL import Image
import io

def plot_dataset(data, title="Dataset Visualization"):
    """
    Plot a 2D dataset with different colors for each class
    
    Args:
        data: numpy array of shape (n_samples, 3) containing (x, y, label)
        title: string for the plot title
    """
    # Split data into features and labels
    X = data[:, :2]
    y = data[:, 2]
    
    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='orange', alpha=0.5, label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', alpha=0.5, label='Class 1')
    
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set equal aspect ratio to maintain circular shapes
    plt.axis('equal')
    
    return plt


def plot_decision_boundary(wMat, aVec, nInput, nOutput, data):
    """
    Plot the decision boundary with probability gradients and test data points
    
    Args:
        wMat: weight matrix
        aVec: activation vector
        nInput: number of input neurons
        nOutput: number of output neurons
        data: array containing test inputs and one-hot encoded labels
    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    import seaborn as sns

    
    # Create a mesh grid for visualization
    x_min, x_max = min(data[:, 0].min(), -5.0), max(data[:, 0].max(), 5.0)
    y_min, y_max = min(data[:, 1].min(), -5.0), max(data[:, 1].max(), 5.0)
    h = 0.05  # Step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Prepare grid points as input
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions and probabilities for each grid point
    grid_points_jax = jnp.array(grid_points)
    probs = act(wMat, aVec, nInput, nOutput, grid_points_jax)
    # probabilities = softmax(np.array(logits), axis=1)
    probabilities = probs / jnp.sum(probs, axis=1, keepdims=True) # simple 
    
    # Use class 1 probability to create a continuous colormap
    Z = probabilities[:, 1].reshape(xx.shape)

    # Set up the plot with seaborn style
    # plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot decision boundary with probability gradients
    contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 50),
                         cmap='RdYlBu_r', alpha=0.7)
    
    # Plot test data points
    test_inputs = data[:, :2]
    test_labels = data[:, 2]
    
    sns.scatterplot(x=test_inputs[test_labels == 0, 0], 
                   y=test_inputs[test_labels == 0, 1],
                   color='blue', marker='o', label='Class 0', 
                   alpha=0.8, s=100, ax=ax)
    sns.scatterplot(x=test_inputs[test_labels == 1, 0], 
                   y=test_inputs[test_labels == 1, 1],
                   color='red', marker='o', label='Class 1', 
                   alpha=0.8, s=100, ax=ax)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Decision Boundary with Probability Gradients')
    ax.legend()
    
    # Add a colorbar with probability labels
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Probability of Class 1')
    
    return fig, ax
    

# Example usage:
# if __name__ == "__main__":
#     from datagen import DataGenerator
    
#     # Create generator and generate some data
#     generator = DataGenerator(train_size=1000, test_size=100, noise_level=0.5, batch_size=10)
    
#     # Generate and plot each type of dataset
#     datasets = {
#         "Circle": 0,
#         "XOR": 1,
#         "Gaussian": 2,
#         "Spiral": 3
#     }
    
#     for name, choice in datasets.items():
#         train_data, _ = generator.generate_random_dataset(choice=choice)
#         plt = plot_dataset(train_data, f"{name} Dataset")
#         plt.show()