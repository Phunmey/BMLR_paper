import pandas as pd
import matplotlib.pyplot as plt


def plot_f1_scores(df):
    unique_dataclasses = sorted(df['dataclass'].unique())
    unique_scenarios = sorted(df['scenario'].unique())
    unique_samples = sorted(df['num_samples'].unique())

    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']

    fig, axes = plt.subplots(nrows=len(unique_dataclasses), ncols=len(unique_scenarios), figsize=(10, 6), sharex='all',
                             sharey='all')
    fig.suptitle(f'Imbalance Scenarios', fontsize=14, y=0.95, ha='center', va='top')

    handles, labels = None, None
    for i, dataclass in enumerate(unique_dataclasses):
        for j, scenario in enumerate(unique_scenarios):
            ax = axes[i, j] if len(unique_dataclasses) > 1 else axes[j]

            subset = df[(df['dataclass'] == dataclass) & (df['scenario'] == scenario)]

            for k, sample in enumerate(unique_samples):
                sample_subset = subset[subset['num_samples'] == sample]
                style = line_styles[k % len(line_styles)]

                ax.plot(sample_subset['sigma'], sample_subset['f1_score'], label=f'{sample}',
                        linestyle=style)

            # Capture legend handles and labels from the first plot
            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()
            # Set title for only the top row
            if i == 0:
                ax.set_title(f'{scenario}', fontsize=13)
            # Set y-axis for only first column
            if j == 0:
                ax.set_ylabel(f'{dataclass} Classes', fontsize=13)
            else:
                ax.tick_params(axis='y', length=0)
            if i == 1:
                ax.tick_params(axis='x', labelsize=10, pad=5)
            else:
                ax.tick_params(axis='x', length=0)

            ax.set_xlabel('')
            ax.set_ylim(0, 1)
    # Set common labels
    fig.text(0.5, 0.04, 'Sigma Values', ha='center', va='center', fontsize=14)
    fig.text(0.02, 0.5, 'F1-Scores', ha='center', va='center', rotation='vertical', fontsize=14)

    if handles and labels:
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.01, 0.5), title='Samples',
                   fontsize=10, title_fontsize=13, frameon=False)

    plt.tight_layout(rect=(0.03, 0.07, 0.9, 0.95))

    plt.savefig("prior_sensitivity_plot.pdf", bbox_inches='tight')

    plt.clf()


if __name__ == '__main__':
    file_path = "sim_test_prior_sensitivity_results.csv"
    df1 = pd.read_csv(file_path, sep='\t', header=0)
    df1 = df1.drop(['data_type', 'model', 'iteration', 'time_taken'], axis=1)
    sigma_mapping = {0.01: '0.01', 0.50: '0.5', 1.00: '1', 5.00: '5', 10.00: '10', 100.00: '100', 1000.00: '1000', }

    df = df1.copy()
    df.loc[:, 'sigma'] = df['sigma'].map(sigma_mapping)

    plot_f1_scores(df)
