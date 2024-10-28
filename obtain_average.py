import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data():
    read_file = pd.read_csv(r'.\results\student_test_result.csv', sep='\t', header=0)

    # for simulated data
    #file_df = read_file.drop(['time_taken', 'flat_conf_mat'], axis=1)  # drop columns
    # group_data = file_df.groupby(['dataclass', 'num_samples', 'scenario', 'data_type',  'model'])[
    #     ['accuracy', 'precision', 'recall', 'f1_score']].agg(['mean']).round(3)

    # for student data
    file_df = read_file.drop(['dataclass', 'num_samples', 'scenario', 'time_taken', 'flat_conf_mat'], axis=1)
    group_data = file_df.groupby(['data_type', 'model'])[
        ['accuracy', 'precision', 'recall', 'f1_score']].agg(['mean']).round(3)

    group_data.to_csv(r".\results\student_test_avg_results.csv")

    # group_data


def metric_barplot(df):
    """
    Obtain a clustered bar plot for simulated data
    """
    data_type = ['ADASYN', 'Imbalanced', 'SMOTE', 'Tomek']
    dataset_mapping = {
        'data_1': (0.35, 0.05, 0.6),
        'data_2': (0.1, 0.2, 0.7),
        'data_3': (0.25, 0.25, 0.5)
    }

    # Replace dataset names with tuples in the dataframe
    df['dataset'] = df['dataset'].map(dataset_mapping)

    for resample in data_type:
        resample_type = df[df['data_type'] == resample]

        # Create the clustered bar plot
        sns.barplot(x='model', y='f1_score', hue='dataset', data=resample_type)

        # Adjust the plot aesthetics
        plt.xlabel('Model')
        plt.ylabel('F1-score')
        if resample == "Imbalanced":
            # Change legend labels
            handles, labels = plt.gca().get_legend_handles_labels()
            labels = [str(dataset_mapping.get(label, label)) for label in labels]
            plt.legend(handles, labels, frameon=False, title='Scenarios',  loc='best')
            plt.tight_layout()
        else:
            plt.legend().remove()

        #plt.savefig(rf"./plots/{resample}_sim.pdf", bbox_inches='tight')

        plt.clf()


def student_barplot(df):
    """
    Obtain a clustered bar plot for real student data
    """
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
    fig.text(0.5, 0.01, 'Metrics', ha='center', va='center', fontsize=20)
    fig.text(0.01, 0.5, 'Scores', ha='center', va='center', rotation='vertical', fontsize=20)

    # Add a single legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Model')

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.03, 0.85, 0.97])
    #plt.show()


    plt.savefig(r".\plots\student_perfomancemetric_plot.pdf", bbox_inches='tight')



if __name__ == '__main__':
    #read_data()
    df = pd.read_csv(r".\results\student_test_avg_results.csv")
    #metric_barplot(df)
    student_barplot(df)
