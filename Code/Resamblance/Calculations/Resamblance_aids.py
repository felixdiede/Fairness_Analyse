import sys
sys.path.append("../Metrics")

import pandas as pd
from Resamblance_Metrics import *

cat_features = ["trt", "hemo", "homo", "drugs", "oprior", "z30", "race", "gender", "str2", "strat", "symptom", "treat", "offtrt", "infected"]
num_features = ["time", "age", "wtkg", "karnof", "preanti", "cd40", "cd420", "cd80", "cd820"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Results_Analysis/.venv/Data/real/aids_original.csv")
aids_synthetic_tabfairgan = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Results_Analysis/.venv/Data/synthetic/aids_synthetic_tabfairgan.csv")

synthetic_data = aids_synthetic_tabfairgan


def statistical_tests(real_data, synthetic_data):
    print_results_t_test(real_data, synthetic_data, attribute=cat_features)
    print_results_mw_test(real_data, synthetic_data, attribute=cat_features)
    print_results_ks_test(real_data, synthetic_data, attribute=cat_features)
    print_results_chi2_test(real_data, synthetic_data, attribute=num_features)

#statistical_tests(real_data, synthetic_data)


# calculate_and_display_distances(real_data, aids_synthetic_tabfairgan, real_data.columns)

# ppc_matrix(real_data, synthetic_data, num_features)

# normalized_contingency_tables()

data_labelling_analysis(real_data, synthetic_data)