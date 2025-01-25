import matplotlib.pyplot as plt
import numpy as np

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

# Example usage:
if __name__ == "__main__":
    from datagen import DataGenerator
    
    # Create generator and generate some data
    generator = DataGenerator(train_size=1000, test_size=100, noise_level=0.5, batch_size=10)
    
    # Generate and plot each type of dataset
    datasets = {
        "Circle": 0,
        "XOR": 1,
        "Gaussian": 2,
        "Spiral": 3
    }
    
    for name, choice in datasets.items():
        train_data, _ = generator.generate_random_dataset(choice=choice)
        plt = plot_dataset(train_data, f"{name} Dataset")
        plt.show()