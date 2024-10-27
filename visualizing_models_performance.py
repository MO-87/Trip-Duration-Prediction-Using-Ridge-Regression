import matplotlib.pyplot as plt
import seaborn as sns


def visualize(rmse, r2):
    errors_df = {
        'Approach': ['Approach 1', 'Approach 1', 'Approach1', 'Approach 2', 'Approach 2', 'Approach2',
                     'Approach 3', 'Approach 3', 'Approach3'],
        'Dataset': ['Train', 'Validation', 'Test', 'Train', 'Validation', 'Test', 'Train', 'Validation', 'Test'],
        'RMSE': rmse,
        'R2': r2
    }

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Dataset', y='RMSE', hue='Approach', marker='o', data=errors_df)
    plt.title('RMSE Across Datasets')
    plt.ylabel('RMSE')
    plt.xlabel('Dataset')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Dataset', y='R2', hue='Approach', marker='o', data=errors_df)
    plt.title('R² Score Across Datasets')
    plt.ylabel('R² Score')
    plt.xlabel('Dataset')
    plt.show()
