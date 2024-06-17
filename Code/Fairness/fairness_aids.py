import pandas as pd
import os
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


os.chdir("/Users/felixdiederichs/PycharmProjects/Results_Analysis/.venv")

# Analyse fairness of aids
aids = pd.read_csv("data/real/aids_original.csv")
aids_synthetic_distcorrgan = pd.read_csv("data/synthetic/aids_synthetic_distcorrgan.csv")
aids_synthetic_tabfairgan = pd.read_csv("data/synthetic/aids_synthetic_tabfairgan.csv")

def calculate_fairness(data, report_path):
    X = data.drop("infected", axis=1)
    y = data["infected"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    fairness_report = report.compare(test_data=X_test,
                                     targets=y_test,
                                     protected_attr=X_test["race"],
                                     models= model,
                                     skip_performance=True
                                     )
    fairness_report.to_html(report_path)

calculate_fairness(aids, "Reports/Fairness_Reports/fairness_report_aids_original.html")


