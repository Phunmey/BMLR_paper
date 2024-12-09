import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from subset_numeric_data import preprocess_student
from calc_vif import iterative_vif_selection

sns.set_palette(palette="deep")
sns_c = sns.color_palette(palette="deep")


def read_data():
    filepath = "data.csv"

    if not os.path.isfile(filepath):
        preprocess_student()

    readData = pd.read_csv(filepath)

    df_x = readData.iloc[:, :-1]
    encode_y = pd.Categorical(readData['Status'])
    df_y = encode_y.codes.astype('int')
    categ = encode_y.categories.astype('object')
    classes = ['Enrolled', 'Graduate']
    get_ind = [categ.get_loc(cls) for cls in classes]
    class_names = categ.take(get_ind)

    return df_x, df_y, class_names


def exploratory_analysis(df_x, df_y):
    """
    This function performs exploratory analysis on the data.
    """
    # heatmap for numerical variables
    plt.subplots(figsize=(10, 8))
    sns.heatmap(df_x.corr(), annot=True, fmt='.2f')
    plt.title("Correlation matrix between all numerical variables", y=1, size=16)
    plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize=14)
    plt.savefig("numvar_heatmap_novif.pdf")

    sns.pairplot(
        data=pd.concat([df_x, pd.DataFrame(df_y, columns=['Status'])], axis=1),
        hue='Status',
        kind="scatter",
        height=2,
        plot_kws={"color": sns_c[1]},
        diag_kws={"color": sns_c[2]}
    )
    plt.title("Pairwise relationship between all numerical variables", y=1, size=16)
    plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize=14)
    plt.savefig("numvar_pairplot_novif.pdf")


def obtain_vif_data(df_x):
    x_vif = iterative_vif_selection(df_x, threshold=10)
    return x_vif


def vif_exploratory_analysis(x_vif, df_y):
    """
    This function performs exploratory analysis on the data.
    """
    # heatmap for numerical variables
    plt.subplots(figsize=(10, 8))
    sns.heatmap(x_vif.corr(), annot=True, fmt='.2f')
    plt.title("Correlation matrix between all numerical variables", y=1, size=16)
    plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize=14)
    plt.savefig("numvar_heatmap_vif.pdf")

    sns.pairplot(
        data=pd.concat([x_vif, pd.DataFrame(df_y, columns=['Status'])], axis=1),
        hue='Status',
        kind="scatter",
        height=2,
        plot_kws={"color": sns_c[1]},
        diag_kws={"color": sns_c[2]}
    )
    plt.title("Pairwise relationship between all numerical variables", y=1, size=16)
    plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize=14)
    plt.savefig("numvar_pairplot_vif.pdf")

