# Evaluating the Impact of Imbalanced Datasets on Binary Classification: A Comparative Study of Resampling Techniques

**Álvaro V. P. M. BANDEIRA  and  Jorge Luis BAZÁN**

This project is an experimental study on the impact of resampling techniques in binary classification for imbalanced datasets. The objective is to explore how various methods of handling class imbalance affect the performance of machine learning models.

# Abstract
Imbalanced datasets challenge traditional machine learning algorithms, especially in real-world applications like medical
diagnosis and fraud detection. This study explores the impact of artificially induced imbalances on binary classification
tasks, focusing on resampling techniques. Through a comprehensive methodology involving various algorithms, including Decision Tree, Random Forest, KNN, SVM, Logistic Regression, and LDA, we observe a universal enhancement in
performance with resampling. Hybrid techniques, particularly SMOTEENN, demonstrate consistent superiority across
imbalance levels. Undersampling techniques show commendable performance in moderate imbalance but decline in high
imbalances. The findings emphasize the efficacy of resampling methods in improving algorithmic performance under varying degrees of imbalance, providing insights for future research in data analysis and classification methodologies.

## File Descriptions

- **`data_simulation.py`**: This script simulates a dataset based on a bivariate normal distribution. While it provides a foundation for experimentation, you can easily replace this simulated data with other datasets—whether synthetic or real-world. For the purpose of this study, it's crucial that the classes are balanced during this simulation phase.

- **`imbalancer.py`**: This module includes functions that artificially introduce different levels of class imbalance into the dataset. You can adjust the imbalance ratio according to your experimentation needs.

- **`resampling.py`**: Here, you'll find various resampling techniques implemented to address class imbalance. The script is designed to be extensible, so feel free to incorporate additional methods that you would like to test.

- **`evaluation.py`**: This file is dedicated to evaluating the outcomes of the experiments. It assesses multiple machine learning algorithms using various evaluation metrics. As with the other components, you are encouraged to add new algorithms and metrics to broaden the scope of your analysis.
- **`fullexperiment.ipynb`**: **COMING SOON**. Full experiment in a Jupyter Notebook with more detailed explanations.

## Getting Started

To get started with this project, clone the repository and ensure you have the necessary libraries installed. Follow the instructions in each file to run simulations, apply resampling techniques, and evaluate the results. Happy experimenting!
