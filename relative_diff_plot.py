import pandas as pd
import matplotlib.pyplot as plt


def read_and_process_data(file_path, is_simulated=True):
    df = pd.read_csv(file_path, sep='\t', header=0)

    # Drop columns based on the type of data
    if is_simulated:
        df = df.drop(['time_taken'], axis=1)
        group_columns = ['dataclass', 'num_samples', 'scenario', 'data_type', 'model']
        output_path = r"sim_test_avg_results.csv"
    else:
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


def sim_relative_diff(df):
    """
    Obtain a relative difference barplot for the simulation data for each sample size and class.
    The baseline model is BMLR. Other models are then compared to the baseline.
    negative bars mean the baseline perform better while positive bars mean otherwise.
    The x-axes are the performance metrics and the clustered bars are the models.
    """
    sample_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
    classes = [3, 5]
    data_type = ['Imbalanced']

    dataset_mapping = {
        'Moderate': 'Moderate',
        'Extreme': 'Extreme',
        'EqualMinorities': 'Equal\nMinorities'}

    df = df.copy()
    df.loc[:, 'scenario'] = df['scenario'].map(dataset_mapping)

    # Extract unique data types and metrics
    scenario_types = df['scenario'].unique()
    # Choose BMLR as the baseline model
    baseline_model = 'BMLR'

    # Define hatch patterns for the bars
    hatch_patterns = ['/', '-', 'x', 'o']
    for resample in data_type:
        # Create the 3 by 3 subplot with shared x-axis and y-axis labels
        fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(24, 12), sharex='all', sharey='all')

        # Initialize variables for legend handles and labels
        handles, labels = None, None
        for i, n_classes in enumerate(classes):
            for j, sample_size in enumerate(sample_sizes):
                # Select the current axis
                ax = axes[i, j]

                df_subset = df[
                    (df['dataclass'] == n_classes) & (df['num_samples'] == sample_size) & (df['data_type'] == resample)]
                baseline_values = df_subset[df_subset['model'] == baseline_model].set_index('scenario')['f1_score']
                models = df_subset['model'].unique()

                # Initialize a DataFrame to hold relative differences
                relative_differences = pd.DataFrame(index=models, columns=scenario_types)

                # Calculate relative difference for each model with respect to the baseline
                for model in models:
                    if model == baseline_model:
                        continue
                    for scenario in scenario_types:
                        model_value = df_subset.loc[(df_subset['model'] == model) &
                                                    (df_subset['scenario'] == scenario), 'f1_score']
                        baseline_value = baseline_values.get(scenario)

                        if not model_value.empty and baseline_value:
                            relative_differences.loc[model, scenario] = (model_value.iloc[
                                                                             0] - baseline_value) / baseline_value
                relative_differences *= 100
                bars = relative_differences.T.plot(kind='bar', ax=ax, width=0.8, legend=False, grid=False)

                # Apply hatches to bars
                for k, bar_container in enumerate(bars.containers):
                    for bar in bar_container:
                        bar.set_hatch(hatch_patterns[k % len(hatch_patterns)])

                # Capture legend handles and labels from the first plot
                if handles is None and labels is None:
                    handles, labels = ax.get_legend_handles_labels()
                # Set title for only the top row
                if i == 0:
                    ax.set_title(f'{sample_size} Samples', fontsize=20)
                # Set y-axis for only first column
                if j == 0:
                    ax.set_ylabel(f'{n_classes} Classes', fontsize=20)
                else:
                    ax.tick_params(axis='y', length=0)
                if i == 1:
                    ax.tick_params(axis='x', labelrotation=40, labelsize=15, pad=5)
                else:
                    ax.tick_params(axis='x', length=0)

                ax.set_xlabel('')
        # Set common labels
        fig.text(0.5, 0.04, 'Imbalance Ratios', ha='center', va='center', fontsize=20)
        fig.text(0.02, 0.5, 'Relative Difference in F1-Score (in %)', ha='center', va='center', rotation='vertical',
                 fontsize=20)

        if handles and labels:
            fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.003, 0.5), title='Model',
                       fontsize=13, title_fontsize=13, frameon=False)

        plt.tight_layout(rect=(0.02, 0.06, 0.95, 0.95))

        plt.savefig(rf".sim_relative_diff_plot.pdf", bbox_inches='tight')

        plt.clf()


if __name__ == '__main__':
    read_and_process_data(r'sim_test_results.csv', is_simulated=True)
    read_and_process_data(r'student_test_result.csv', is_simulated=False)

    # Read processed data for plotting
    student_avg = pd.read_csv(r"student_test_avg_results.csv", header=0)
    df_filtered = student_avg[~student_avg["model"].str.contains("LogReg", na=False)]
    student_relative_diff(df_filtered)

    # Read processed data for plotting
    sim_avg = pd.read_csv(r"sim_test_avg_results.csv", header=0)
    df_filtered = sim_avg[~sim_avg["model"].str.contains("LogReg", na=False)]
    sim_relative_diff(df_filtered)


