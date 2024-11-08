import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette(palette="deep")
sns_c = sns.color_palette(palette="deep")


def preprocess_student():
    datapath = r"\data"
    read_data = pd.read_csv(f"{datapath}/students_dropout.csv")
    read_data.rename(columns={'day_even_attendance': 'Attendance', 'Educational special needs': 'Special_needs',
                              'Tuition fees up to date': 'Tuition', 'Scholarship holder': 'Scholarship',
                              'Age at enrollment': 'Age',
                              'Curricular units 1st sem (credited)': 'CU1_credited',
                              'Curricular units 1st sem (enrolled)': 'CU1_enrolled',
                              'Curricular units 1st sem (evaluations)': 'CU1_eval',
                              'Curricular units 1st sem (approved)': 'CU1_approved',
                              'Curricular units 1st sem (without evaluations)': 'CU1_weval',
                              'Curricular units 1st sem (grade)': 'CU1_grade',
                              'Curricular units 2nd sem (credited)': 'CU2_credited',
                              'Curricular units 2nd sem (enrolled)': 'CU2_enrolled',
                              'Curricular units 2nd sem (evaluations)': 'CU2_eval',
                              'Curricular units 2nd sem (approved)': 'CU2_approved',
                              'Curricular units 2nd sem (without evaluations)': 'CU2_weval',
                              'Curricular units 2nd sem (grade)': 'CU2_grade',
                              'Previous qualification (grade)': 'Previous_grade', 'Admission grade': 'Admission_grade',
                              'Unemployment rate': 'Unemployment_rate', 'Inflation rate': 'Inflation_rate',
                              'Target': 'Status'},
                     inplace=True)

    read_data['CU_enrolled'] = read_data['CU1_enrolled'] + read_data['CU2_enrolled']
    read_data['CU_grade'] = read_data['CU1_grade'] + read_data['CU2_grade']
    read_data['CU_approved'] = read_data['CU1_approved'] + read_data['CU2_approved']
    read_data['CU_weval'] = read_data['CU1_weval'] + read_data['CU2_weval']
    read_data['CU_eval'] = read_data['CU1_eval'] + read_data['CU2_eval']
    read_data['CU_credited'] = read_data['CU1_credited'] + read_data['CU2_credited']

    df_num = read_data.drop(columns=['CU1_enrolled', 'CU1_credited', 'CU1_grade', 'CU1_approved', 'CU1_weval',
                                     'CU1_eval', 'CU2_enrolled', 'CU2_credited', 'CU2_grade', 'CU2_approved',
                                     'CU2_weval', 'CU2_eval'])
    # select numerical variables
    df_x = df_num[['Age', 'CU_enrolled', 'CU_credited', 'CU_grade', 'CU_approved', 'CU_weval', 'CU_eval',
                   'Unemployment_rate', 'Inflation_rate', 'GDP', 'Previous_grade', 'Admission_grade']]

    codes, uniques = pd.factorize(read_data['Status'], sort=True)
    df_y = pd.Categorical(read_data['Status']).codes

    class_names = uniques[1:]

    return df_x, df_y, class_names


def exploratory_analysis(df_x, df_y):
    """
    This function performs exploratory analysis on the data.
    """
    plt.subplots(figsize=(10, 8))
    sns.heatmap(df_x.corr(), annot=True, fmt='.2f')
    plt.title("Correlation matrix between all numerical variables", y=1, size=16)
    plt.xticks(rotation=45, horizontalalignment='right', fontweight='light', fontsize=14)
    plt.savefig("numvar_heatmap.pdf")

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
    plt.savefig("numvar_pairplot.pdf")

    return
