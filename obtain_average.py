import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_and_process_data(file_path, is_simulated=True):
    df = pd.read_csv(file_path, sep='\t', header=0)

    # Drop columns based on the type of data
    if is_simulated:
        df = df.drop(['time_taken', 'flat_conf_mat'], axis=1)
        group_columns = ['dataclass', 'num_samples', 'scenario', 'data_type', 'model']
        output_path = r".sim_test_avg_results.csv"
    else:
        df = df.drop(['dataclass', 'num_samples', 'scenario', 'time_taken', 'flat_conf_mat'], axis=1)
        group_columns = ['data_type', 'model']
        output_path = r"student_test_avg_results.csv"

    grouped_df = df.groupby(group_columns)[['accuracy', 'precision', 'recall', 'f1_score']].agg(['mean']).round(3)

    grouped_df.to_csv(output_path)


def metric_barplot(df):
    sample_sizes = [2000, 5000, 10000]
    classes = [3, 5, 10]
    data_type = ['Imbalanced', 'ADASYN', 'SMOTE', 'Tomek']

    dataset_mapping = {1: 'Moderate', 2: 'Extreme', 3: 'Equal\nMinorities'}

    # Replace scenarios 1, 2, and 3 in the dataframe
    df['scenario'] = df['scenario'].map(dataset_mapping)
    for resample in data_type:
        # Create the 3 by 3 subplot with shared x-axis and y-axis labels
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), sharex='all', sharey='all')

        # Add a common title for all columns
        fig.suptitle(f'{resample} Data', fontsize=22, y=0.95)  # , ha='center', va='top')

        # Initialize variables for legend handles and labels
        handles, labels = None, None
        for i, n_classes in enumerate(classes):
            for j, sample_size in enumerate(sample_sizes):
                df_subset = df[
                    (df['dataclass'] == n_classes) & (df['num_samples'] == sample_size)]
                resample_type = df_subset[df_subset['data_type'] == resample]

                # Select the current axis
                ax = axes[i, j]
                resample_type = resample_type.set_index('scenario')
                # Create the clustered bar plot on the current axis
                sns.barplot(x='scenario', y='f1_score', hue='model', data=resample_type,
                            width=0.8, ax=ax, legend=(i == 0 and j == 0))

                # Capture the legend handles and labels from the first plot
                if i == 0 and j == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend_.remove()  # Remove the individual legend from the first subplot

                # Set title for only the top row
                if i == 0:
                    ax.set_title(f'{sample_size} Samples', fontsize=20)
                # Set y-axis for only first column
                if j == 0:
                    ax.set_ylabel(f'{n_classes} Classes', fontsize=15)
                if i == 2:
                    ax.tick_params(axis='x', rotation=10, labelsize=12)

                ax.set_xlabel('')
        # Set common labels
        fig.text(0.5, 0.04, 'Scenarios', ha='center', va='center', fontsize=20)
        fig.text(0.02, 0.5, 'F1-Score', ha='center', va='center', rotation='vertical', fontsize=20)

        if handles and labels:
            fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.003, 0.5), title='Model',
                       fontsize=13, title_fontsize=13, frameon=False)

        plt.tight_layout(rect=[0.04, 0.06, 0.91, 0.95])

        plt.savefig(rf"{resample}_sim.pdf", bbox_inches='tight')

        plt.clf()


def student_barplot(df):
    ax = None
    # Extract unique data types and models
    data_types = df['data_type'].unique()
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Create the 2 by 2 subplot with shared x-axis and y-axis labels
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10), sharex='all', sharey='all')
    axes = axes.flatten()

    # Iterate over each data type and create a clustered bar plot for each
    for i, data_type in enumerate(data_types):
        subset = df[df['data_type'] == data_type]

        # Plotting
        ax = axes[i]
        subset = subset.set_index('model')[metrics]
        subset.T.plot(kind='bar', ax=ax, width=0.8, legend=False)

        # Set the title
        ax.set_title(f'{data_type}', fontsize=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=0, labelsize=15)  # Ensure x-ticks are horizontal

    # Set common labels
    fig.text(0.5, 0.04, 'Metrics', ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, 'Scores', ha='center', va='center', rotation='vertical', fontsize=20)

    # Add a single legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.003, 0.5), title='Model',
               fontsize=13, title_fontsize=13, frameon=False)

    # Adjust layout
    plt.tight_layout(rect=[0.04, 0.06, 0.91, 0.95])

    plt.savefig(r"student_performancemetric_plot.pdf",
                bbox_inches='tight')


if __name__ == '__main__':
    read_and_process_data(r'sim_test_results.csv', is_simulated=True)
    read_and_process_data(r'student_test_result.csv', is_simulated=False)

    # Read processed data for plotting (adjust dataframe header before reading)
    df_simulated = pd.read_csv(r"sim_test_avg_results.csv")
    metric_barplot(df_simulated)

    df_student = pd.read_csv(r"student_test_avg_results.csv")
    student_barplot(df_student)

