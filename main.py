import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


class Clusterer(object):
    def __init__(self, file_name = ""):
        
        self.file_name = file_name
        self.X = self.read_data(file_name)
        self.n_dims = self.X.shape[1]
        self.k = 4 * self.n_dims - 1

    def read_data(self, path):
        df = pd.read_csv(path).drop(columns=["id"])
        X = df.values
        return X

    def cluster_data(self):
        kmeans_model = MiniBatchKMeans(n_clusters=self.k, n_init=100, batch_size=1000, random_state=42)
        
        labels = kmeans_model.fit_predict(self.X)
        return labels


    def save_results(self, labels, output_file="submission.csv"):
        """Save the clustering results into a CSV file."""
        pd.DataFrame({"id": range(len(labels)), "label": labels}).to_csv(output_file, index=False)


if __name__ == "__main__":
    public_clusterer = Clusterer("public_data.csv")
    print(f"Loaded data with shape: {public_clusterer.X.shape}")
    public_labels = public_clusterer.cluster_data()
    public_clusterer.save_results(public_labels, output_file="public_submission.csv")

    private_clusterer = Clusterer("private_data.csv")
    print(f"Loaded private data with shape: {private_clusterer.X.shape}")
    private_labels = private_clusterer.cluster_data()
    private_clusterer.save_results(private_labels, output_file="private_submission.csv")


# python3 main.py
# python3 main.py && python3 eval.py