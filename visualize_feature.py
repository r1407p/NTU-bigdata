import pandas as pd
import os
import matplotlib.pyplot as plt


def read_data(path):
    df = pd.read_csv(path).drop(columns=["id"])
    X = df.values
    return X

def visualzie_feature(X, feature_index1, feature_index2):
    """
    Visualize the relationship between two features in the dataset.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features)
    - feature_index1: index of the first feature to visualize
    - feature_index2: index of the second feature to visualize
    """
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, feature_index1], X[:, feature_index2], alpha=0.5)
    plt.title(f"Feature {feature_index1} vs Feature {feature_index2}")
    plt.xlabel(f"Feature {feature_index1}")
    plt.ylabel(f"Feature {feature_index2}")
    plt.grid()
    plt.show()
    os.makedirs("visualization", exist_ok=True)
    plt.savefig(f"visualization/feature_{feature_index1}_vs_{feature_index2}.png")
    plt.close()

if __name__ == "__main__":
    # Example usage
    path = "public_data.csv"  # Replace with your file path
    X = read_data(path)
    print(X)
    print(f"Data shape: {X.shape}")
    print(f"First few rows:\n{X[:5]}")

    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            print(f"Visualizing feature {i} vs feature {j}")
            visualzie_feature(X, i, j)
            
            