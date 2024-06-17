import pandas as pd
from Metrics.Utility_Metrics import trtr, tstr

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/obesity_generation.csv")

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/obesity")

dataframes = {}

for file_names in os.listdir():
        file_path = os.path.join(file_names)
        dataframes[file_names] = pd.read_csv(file_path)

synthetic_data = dataframes[""]

trtr(real_data, synthetic_data, "NObeyesdad")

tstr(real_data, synthetic_data, "NObeyesdad")
