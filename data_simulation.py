import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set a random seed for reproducibility
np.random.seed(0)

# Generate random data for class 0
class_0_features = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], size=1500)
class_0_labels = np.zeros(1500, dtype=int)

# Generate random data for class 1
class_1_features = np.random.multivariate_normal([4, 3], [[1, 0.5], [0.5, 1]], size=1500)
class_1_labels = np.ones(1500, dtype=int)

# Combine the data
features = np.vstack([class_0_features, class_1_features])
labels = np.concatenate([class_0_labels, class_1_labels])

# Create a DataFrame
data = pd.DataFrame({'Feature_1': features[:, 0], 'Feature_2': features[:, 1], 'Label': labels})

X_train, X_test, y_train, y_test = train_test_split(data.drop(["Label"],axis=1), data.Label, test_size=1/3, random_state=42, stratify=data.Label)

data = pd.concat([X_train,y_train],axis=1)
