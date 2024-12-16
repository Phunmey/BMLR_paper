import pandas as pd
import matplotlib.pyplot as plt


def read_and_process_data(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0)

    df = df.drop(['dataclass', 'num_samples', 'scenario', 'time_taken'], axis=1)
    group_columns = ['data_type', 'model']
    output_path = r"student_test_avg_results.csv"

    grouped_df = df.groupby(group_columns)[['accuracy', 'bal_accuracy', 'precision', 'recall', 'f1_score']].agg(
        ['mean']).round(3)

    grouped_df.to_csv(output_path)


def student_relative_diff(df):
    """
    Obtain a relative difference barplot for the student data for each data-type.
    The baseline model is BMLR. Other models are then compared to the baseline.
    negative bars mean the baseline perform better while positive bars mean otherwise.
    The x-axes are the performance metrics and the clustered bars are the models.
    """
    ax = None

    # Extract unique data types and metrics
    data_types = df['data_type'].unique()
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Choose BMLR as the baseline model
    baseline_model = 'BMLR'

    # Define hatch patterns for the bars
    hatch_patterns = ['/', '-', 'x', 'o']

    # Initialize the figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex='all', sharey='all')
    axes = axes.flatten()

    # Iterate over each data type and calculate relative differences
    for i, data_type in enumerate(data_types):
        subset = df[df['data_type'] == data_type]
        baseline_values = subset[subset['model'] == baseline_model][metrics].values.flatten()

        # Initialize a DataFrame to hold relative differences
        relative_differences = pd.DataFrame(index=subset['model'].unique(), columns=metrics)

        for metric in metrics:
            relative_differences[metric] = (subset.set_index('model')[metric] - baseline_values[
                metrics.index(metric)]) / baseline_values[metrics.index(metric)]

        # Plotting
        ax = axes[i]
        relative_differences *= 100
        bars = relative_differences.T.plot(kind='bar', ax=ax, width=0.8, legend=False, grid=False)

        # Apply hatches to bars
        for j, bar_container in enumerate(bars.containers):
            for bar in bar_container:
                bar.set_hatch(hatch_patterns[j % len(hatch_patterns)])

        # Set the title
        ax.set_title(f'{data_type}', fontsize=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=0, labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # Customization: Remove ticks
        if i < 2:  # First row
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
        if i % 2 == 1:  # Second column
            ax.tick_params(axis='y', left=False, labelleft=False)

    # Set common labels
    fig.text(0.5, 0.02, 'Metrics', ha='center', va='center', fontsize=20)
    fig.text(0.02, 0.5, 'Relative Difference (in %)', ha='center', va='center', rotation='vertical', fontsize=20)

    # Add a single legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.97, 0.5), title='Model',
               fontsize=13, title_fontsize=13, frameon=False)

    # Adjust layout
    plt.tight_layout(rect=(0.03, 0.03, 0.87, 0.97))

    plt.savefig(r"student_relative_diff_plot.pdf")


if __name__ == '__main__':
    read_and_process_data(r'student_test_result.csv')

    # Read processed data for plotting
    student_avg = pd.read_csv(r"student_test_avg_results.csv", header=0)
    student_relative_diff(student_avg)
