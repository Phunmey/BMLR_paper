import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from subset_numeric_data import preprocess_student
from calc_vif import iterative_vif_selection

sns.set_palette(palette="deep")
sns_c = sns.color_palette(palette="deep")


def read_data(filepath="data.csv", classes=None):
    if classes is None:
        classes = ['Enrolled', 'Graduate']
    if not os.path.isfile(filepath):
        preprocess_student()
    readData = pd.read_csv(filepath)

    df_x = readData.iloc[:, :-1]
    encode_y = pd.Categorical(readData['Status'])
    df_y = encode_y.codes.astype('int')
    categ = encode_y.categories.astype('object')
    class_names = categ.take([categ.get_loc(cls) for cls in classes])
    return df_x, df_y, class_names


def plot_heatmap(df, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f')
    plt.title("Correlation Matrix", y=1, size=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.savefig(filename)


def plot_pairplot(df_x, df_y, filename):
    combined_df = pd.concat([df_x, pd.DataFrame(df_y, columns=['Status'])], axis=1)
    sns.pairplot(
        data=combined_df,
        hue='Status',
        kind="scatter",
        height=2,
        plot_kws={"color": sns_c[1]},
        diag_kws={"color": sns_c[2]}
    )
    plt.savefig(filename)


def exploratory_analysis(df_x, df_y, prefix="novif"):
    plot_heatmap(df_x, f"numvar_heatmap_{prefix}.pdf")
    plot_pairplot(df_x, df_y, f"numvar_pairplot_{prefix}.pdf")


def obtain_vif_data(df_x, threshold=10):
    return iterative_vif_selection(df_x, threshold)


def vif_exploratory_analysis(x_vif, df_y):
    exploratory_analysis(x_vif, df_y, prefix="vif")
