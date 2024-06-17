import sys
sys.path.append("../Metrics")

import pandas as pd
from Resamblance_Metrics import *
from glob import glob

import os

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)



cat_features = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"]
num_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/obesity_generation.csv")

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/obesity")

dataframes = {}

for file_names in os.listdir():
        file_path = os.path.join(file_names)
        dataframes[file_names] = pd.read_csv(file_path)

synthetic_data = dataframes["obesity_tvae_1000.csv"]


def statistical_tests(real_data, synthetic_data):
        # print_results_t_test(real_data, synthetic_data, num_features)
        # print_results_mw_test(real_data, synthetic_data, num_features)
        # print_results_ks_test(real_data, synthetic_data, num_features)
        print_results_chi2_test(real_data, synthetic_data, cat_features)

statistical_tests(real_data, synthetic_data)

calculate_and_display_distances(real_data, synthetic_data, num_features)

ppc_matrix(real_data, synthetic_data, num_features)

# normalized_contingency_tables(real_data, dataframes["obesity_ctgan_500.csv"], cat_features)

data_labelling_analysis(real_data, synthetic_data)


