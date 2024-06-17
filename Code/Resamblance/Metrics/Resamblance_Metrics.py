import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy import stats
from scipy.stats import entropy, wasserstein_distance, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning


"""
Chapter 1: Univariate Resemblance Analysis
    1.1 Statistical test for numerical attributes
        1.1.1 Student T-test for the comparison of means
        1.1.2 Mann Whitney U-test for population comparison
        1.1.3 Kolmogorov-Smirnov test for distribution comparison
"""
def t_test(real_data, synthetic_data, attribute, alpha=0.05):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    t_statistic, p_value = stats.ttest_ind(vector1, vector2, alternative="two-sided")

    if p_value < alpha:
        conclusion = 0
            #"H0 is rejected. For this attribute, the resamblance is not given."
    else:
        conclusion = 1
            #"H0 is not rejected. For this attribute, the resembalance is given."

    return t_statistic, p_value, conclusion

def print_results_t_test(real_data, synthetic_data, attribute):

    results = {}
    for attr in attribute:
        t_statistic, p_value, conclusion = t_test(real_data, synthetic_data, attr)

        results[attr] = {
            "t-Statistic": t_statistic,
            "p-Value": p_value,
            "Conclusion": conclusion
        }

    results_df = pd.DataFrame(results).transpose()
    """print(results_df.to_markdown(numalign="left", stralign="left"))"""





def mw_test(real_data, synthetic_data, attribute, alpha=0.05):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    mw_statistic, p_value = stats.mannwhitneyu(vector1, vector2, alternative="two-sided")

    if p_value < alpha:
        conclusion = 0
            #"H0 is rejected. For this attribute, the resamblance is not given."
    else:
        conclusion = 1
            #"H0 is not rejected. For this attribute, the resembalance is given."

    return mw_statistic, p_value, conclusion

def print_results_mw_test(real_data, synthetic_data, attribute):


    results = {}
    for attr in attribute:
        mw_statistic, p_value, conclusion = mw_test(real_data, synthetic_data, attr)

        results[attr] = {
            "mw-Statistic": mw_statistic,
            "p-Value": p_value,
            "Conclusion": conclusion
        }

    results_df = pd.DataFrame(results).transpose()
    """print(results_df.to_markdown(numalign="left", stralign="left"))"""





def ks_test(real_data, synthetic_data, attribute, alpha=0.05):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    ks_statistic, p_value = stats.ks_2samp(vector1, vector2, alternative = "two-sided")

    if p_value < alpha:
        conclusion = 0
            #"H0 is rejected. For this attribute, the resamblance is not given."
    else:
        conclusion = 1
            #"H0 is not rejected. For this attribute, the resembalance is given."

    return ks_statistic, p_value, conclusion

def print_results_ks_test(real_data, synthetic_data, attribute):


    results = {}
    for attr in attribute:
        ks_statistic, p_value, conclusion = ks_test(real_data, synthetic_data, attr)

        results[attr] = {
            "ks-Statistic": ks_statistic,
            "p-Value": p_value,
            "Conclusion": conclusion
        }

    results_df = pd.DataFrame(results).transpose()
    """print(results_df.to_markdown(numalign="left", stralign="left"))"""





"""
Chapter 1: Univariate Resemblance Analysis
    1.2 Statistical test for categorical attributes
        1.2.1 Chi-square test 
"""
def chi2_test(real_data, synthetic_data, attribute, alpha=0.05):
    contingency_table = pd.crosstab(real_data[attribute], synthetic_data[attribute])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    if chi2 < alpha:
        conclusion = 1
            # "H0 is rejected. For this attribute, the resembalance is given."
    else:
        conclusion = 0
            #"H0 is not rejected. For this attribute, the resamblance is not given."

    return chi2, p, dof, expected, conclusion

def print_results_chi2_test(real_data, synthetic_data, attribute):

    results = {}
    for attr in attribute:
        chi_statistic, p_value, _, _, conclusion = chi2_test(real_data, synthetic_data, attr)

        results[attr] = {
            "chi2-Statistic": chi_statistic,
            "p-Value": p_value,
            "Conclusion": conclusion
        }

    results_df = pd.DataFrame(results).transpose()
    """print(results_df.to_markdown(numalign="left", stralign="left"))"""





"""
Chapter 1: Univariate Resemblance Analysis
    1.3 Distance calculation
        1.3.1 Cosine distance
        1.3.2 Jensen-Shannon Distance
        1.3.3 Kullback-Leibler Divergence
        1.3.4 Wassertein Distance
"""
def cos_distance(real_data, synthetic_data, attribute):
    vector1 = np.array([real_data[attribute]])
    vector2 = np.array([synthetic_data[attribute]])

    cos_dist = cosine_distances(vector1, vector2)

    return cos_dist



def js_distance(real_data, synthetic_data, attribute):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])
    js_dist = jensenshannon(vector1, vector2)

    return js_dist



def kl_divergence(real_data, synthetic_data, attribute):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    p = np.histogram(vector1)[0] / len(vector1)
    q = np.histogram(vector2)[0] / len(vector2)

    kl_pq = entropy(p, q)

    return kl_pq



def was_distance(real_data, synthetic_data, attribute):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    u_values = np.histogram(vector1)[0] / len(vector1)
    v_values = np.histogram(vector2)[0] / len(vector2)

    ws_distance = wasserstein_distance(u_values, v_values)

    return ws_distance


import pandas as pd


def calculate_and_display_distances(real_data, synthetic_data, attribute):
    thresholds = {
        "Cosinus": 0.3,
        "Jensen-Shannon": 0.1,
        "KL-Divergenz": 0.1,
        "Wasserstein": 0.3
    }

    distance_functions = {
        "Cosinus": cos_distance,  # Ersetze durch deine tatsächlichen Funktionen
        "Jensen-Shannon": js_distance,
        "KL-Divergenz": kl_divergence,
        "Wasserstein": was_distance
    }

    all_results = {}
    total_true_false = {"true": 0, "false": 0}

    for distance_name, distance_func in distance_functions.items():
        results = []
        for attr in attribute:
            try:
                distance = distance_func(real_data, synthetic_data, attr)
                results.append({"Attribute": attr, "Distance": distance})
            except ValueError:
                results.append({"Attribute": attr, "Distance": "N/A (Error)"})

        df = pd.DataFrame(results)
        df["Conclusion"] = df["Distance"].apply(lambda x: "true" if x < thresholds[distance_name] else "false")

        markdown_table = df.to_markdown(index=False, numalign="left", stralign="left")
        all_results[distance_name] = markdown_table

        """
        # Ausgabe der Tabelle und Zusammenfassung direkt darunter
        print(f"\n## {distance_name} Distanzen\n")
        print(markdown_table)"""

        # Zählen von true/false pro Tabelle
        value_counts = df["Conclusion"].value_counts()
        for value, count in value_counts.items():
            total_true_false[value] += count
            """
            print(f"Anzahl '{value}': {count}")

    # Gesamtzusammenfassung am Ende
    print("\n## Gesamtzusammenfassung\n")
    for value, count in total_true_false.items():
        print(f"Gesamt Anzahl '{value}': {count}")"""
    summary = total_true_false.copy()

    print(summary["true"]/ (summary["true"] + summary["false"]))

"""
Chapter 2: Multivariate Relationship Analysis
    2.1 PPC Matrices comparison
"""

def ppc_matrix(real_data, synthetic_data, num_features):
    threshold = 0.1

    num_real_data = real_data[num_features]
    num_synthetic_data = synthetic_data[num_features]

    corr_matrix_real = num_real_data.corr()
    corr_matrix_syn = num_synthetic_data.corr()

    diff_matrix = np.abs(corr_matrix_real - corr_matrix_syn)

    """print("\n Correlation Difference Matrix\n")
    print(diff_matrix)"""

    diff_matrix = diff_matrix.replace(np.diag(np.ones(diff_matrix.shape[0])), np.nan)

    # Count values below the threshold
    count_below_threshold = (diff_matrix < threshold).sum().sum()
    count_below_threshold = count_below_threshold / 2

    """print(f"\nNumber of values below {threshold}: {count_below_threshold}")"""

    number_of_relations = (len(num_features) * (len(num_features) - 1)) / 2

    """print(f"Number of relations (numerical features): {number_of_relations}")"""

    print(count_below_threshold/ number_of_relations)

"""
Chapter 2: Multivariate Relationship Analysis
    2.2 Normalized contingency tables comparison
"""
"""def normalized_contingency_tables(real_data, synthetic_data, attributes):
    results = {}
    for attr in attributes:
        table = pd.crosstab(real_data[attr], synthetic_data[attr], normalize='all')

        # Calculate the absolute deviation correctly
        expected = np.outer(table.sum(axis=0), table.sum(axis=1)) / table.sum().sum()
        absolute_deviation = np.sum(np.abs(table - expected))

        table_md = table.to_markdown(numalign='left', stralign='left')

        results[attr] = {
            "Contingency tables (Markdown)": table_md,
            "Absolute deviation": absolute_deviation
        }

    # Output the results
    for attr, values in results.items():
        print(f"\nAttribute: {attr}")
        print(values["Contingency tables (Markdown)"])
        print(f"Absolute deviation: {values['Absolute deviation']:.4f}")  # Formatting to 4 decimal places

    return results"""


"""
Chapter 3: DLA 
"""
def data_labelling_analysis(real_data, synthetic_data):
    real_data = pd.get_dummies(real_data)
    synthetic_data = pd.get_dummies(synthetic_data)

    real_data["label"] = 0
    synthetic_data["label"] = 1

    df = pd.concat([real_data, synthetic_data], axis=0)

    df = df.dropna(axis=1)

    # Create a feature dataset and a target dataset
    X = df.drop("label", axis=1)
    y = df["label"]

    """scaler = StandardScaler()
    X = scaler.fit_transform(X)"""

    # Split the data into train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Create classifier
        classifiers = [
            RandomForestClassifier(n_estimators = 100, n_jobs = 3, random_state = 9),
            KNeighborsClassifier(n_neighbors =10, n_jobs = 3),
            DecisionTreeClassifier(random_state = 9),
            SVC(C = 100, max_iter = 300, kernel = "linear", probability = True, random_state = 9),
            MLPClassifier(hidden_layer_sizes = (128,64,32), max_iter = 300, random_state = 9)
        ]

        results = {}
        for clf in classifiers:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results[clf.__class__.__name__] = classification_report(y_test, y_pred, output_dict=True)

    # Ergebnisse in DataFrame umwandeln
    results_df = pd.DataFrame(results).transpose().drop(columns=["macro avg", "weighted avg"])

    average_accuracy = results_df["accuracy"].mean()
    highest_accuracy = max(results_df["accuracy"])
    lowest_accuracy = min(results_df["accuracy"])

    # Markdown-Tabelle erstellen und ausgeben
    """print(results_df.to_markdown(numalign="left", stralign="left", floatfmt=".4f"))"""
    print(highest_accuracy)
    print(average_accuracy)
    print(lowest_accuracy)





