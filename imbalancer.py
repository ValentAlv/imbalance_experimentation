import numpy as np
import pandas as pd

def imbalancer(data, target_column, class_to_remove, removal_percentage):
    np.random.seed(1)
    # Copy the original data to avoid modifying the original DataFrame
    imbalanced_data = data.copy()
    
    # Find indices of the class to be removed
    indices_to_remove = imbalanced_data[imbalanced_data[target_column] == class_to_remove].index
    
    # Calculate the number of samples to remove
    num_samples_to_remove = int(len(indices_to_remove) * removal_percentage)
    
    # Randomly select samples to remove
    samples_to_remove = np.random.choice(indices_to_remove, size=num_samples_to_remove, replace=False)
    
    # Drop the selected samples
    imbalanced_data = imbalanced_data.drop(samples_to_remove)
    
    return imbalanced_data
