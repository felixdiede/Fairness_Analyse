import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import get_column_plot

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data")

real_data = pd.read_csv("real/obesity_generation.csv")
synthetic_data = pd.read_csv("synthetic/obesity_decaf.csv")

synthetic_data.to_csv("synthetic/obesity_synthetic_decaf.csv", index=False)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name="TUE"
)

fig.show()
