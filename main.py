import numpy as np
import pandas as pd
from grader import score


class Clusterer(object):
    def __init__(self, file_name = ""):
        
        self.file_name = file_name
        self.X = self.read_data(file_name)
        self.n_dims = self.X.shape[1]
        self.k = 4 * self.n_dims - 1
        print(self.k)

    def read_data(self, path):
        df = pd.read_csv(path).drop(columns=["id"])
        X = df.values
        return X

if __name__ == "__main__":
    public_clusterer = Clusterer("public_data.csv")
    print(f"Loaded data with shape: {public_clusterer.X.shape}")
    
    # Further processing can be added here