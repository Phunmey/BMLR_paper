"""
Generate simulated data given different classes, imbalanced ratio, and number of samples.
"""


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)


def generate_multiclass_data(n_samples, class_ratios, n_classes, n_features=10, n_repeated=0, dis_variable=(1, 100)):
    np.random.seed(42)  # for reproducibility

    # Ensure n_informative is the same as n_features
    n_informative = np.ceil(n_features / 2).astype(int)
    n_redundant = n_features - n_informative - n_repeated

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        flip_y=0,
        weights=class_ratios,
        random_state=42
    )

    # Discretize the first feature column to represent a discrete variable
    X[:, 0] = np.interp(X[:, 0], (X[:, 0].min(), X[:, 0].max()), dis_variable)
    X[:, 0] = np.round(X[:, 0])

    feature_names = [f'Feature_{i}' for i in range(n_features)]

    df = pd.DataFrame(X, columns=feature_names)
    df['Class'] = y

    return df


# Define the three specific class ratio scenarios
class_ratio_scenarios = [
    {'n_classes': 3, 'class_ratios': [
        (0.5, 0.3, 0.2),  # Moderately Imbalanced
        (0.8, 0.02, 0.18),  # Extremely Imbalanced
        (0.7, 0.15, 0.15)  # Extremely Imbalanced but Equal Minorities
    ]},
    {'n_classes': 5, 'class_ratios': [
        (0.4, 0.2, 0.15, 0.15, 0.1),   # Moderately Imbalanced
        (0.6, 0.1, 0.04, 0.2, 0.06),  # Extremely Imbalanced
        (0.6, 0.1, 0.1, 0.1, 0.1)  # Extremely Imbalanced but Equal Minorities
    ]},
    {'n_classes': 10, 'class_ratios': [
        (0.35, 0.1, 0.1, 0.1, 0.1, 0.08, 0.06, 0.05, 0.04, 0.02),  # Moderately Imbalanced
        (0.7, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02),  # Extremely Imbalanced
        (0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)  # Extremely Imbalanced but Equal Minorities
    ]}
]

sample_sizes = [2000, 5000, 10000]  # Define different sample sizes for each scenario
# feature_counts = [5, 10]  # Define different numbers of features

for scenario in class_ratio_scenarios:
    n_classes = scenario['n_classes']
    class_ratios = scenario['class_ratios']
    print(f"n_classes: {n_classes}, class_ratios: {class_ratios}")
    for scenario_counter, class_ratio in enumerate(class_ratios, start=1):
       for n_samples in sample_sizes:
            class_ratio = np.round(class_ratio, 2)
            df = generate_multiclass_data(n_samples, class_ratio, n_classes)
            print(f"\nScenario {scenario_counter}, Class {n_classes}: Sample Size {n_samples}, Class Ratios {class_ratio}")
            print(np.round(df['Class'].value_counts(normalize=True), 2))

            filename = rf".\simulated_data\sim_class{n_classes}_{n_samples}samples_{scenario_counter}.csv"
            df.to_csv(filename, index=False)
