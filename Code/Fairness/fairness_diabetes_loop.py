import pandas as pd
import os
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
from glob import glob
import styleframe


num_features = ["BMI", "MentHlth", "PhysHlth", "Age"]

os.chdir(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\synthetic\diabetes\distcorrgan")

def custom_sort(file_name):
    if file_name.endswith(".csv"):
        file_name = file_name[:-4]  # Entferne ".csv"

    parts = file_name.split("_")
    model = parts[1]

    # Überprüfen, ob eine Lambda-Angabe vorhanden ist
    if len(parts) == 4:
        epoch = int(parts[2])
        lambda_value = parts[3]
    else:  # Kein Lambda vorhanden
        epoch = int(parts[2])
        lambda_value = ""  # Leerer String für Lambda

    model_priority = {"tabfairgan": 1, "distcorrgan": 2, "multifairgan": 3, "decaf": 4, "TVAE": 5, "CTGAN": 6}
    lambda_priority = {"02": 1, "04": 2, "06": 3, "08": 4, "1": 5, "15": 6, "2": 7, "5": 8}

    return model_priority.get(model, 99), epoch, lambda_priority.get(lambda_value, 99)


file_names = os.listdir()

sorted_files = sorted(file_names, key=custom_sort)

all_data = pd.DataFrame()

for file_path in sorted_files:
    # Dateinamen ohne Erweiterung extrahieren (für den Report-Namen)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    print(file_name)

    data = pd.read_csv(file_path)

    data = pd.get_dummies(data)

    scaler = MinMaxScaler()
    data[num_features] = scaler.fit_transform(data[num_features])

    # Features und Zielvariable trennen
    X = data.drop("Diabetes_binary", axis=1)
    y = data["Diabetes_binary"]

    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell trainieren (hier könnte man verschiedene Modelle ausprobieren)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Fairness-Report erstellen und speichern
    fairness_report = report.compare(
        test_data=X_test,
        targets=y_test,
        protected_attr=X_test["Income"],
        models=model,
        skip_performance=True
    )
    report_data = fairness_report.data

    report_data[file_name] = report_data

    all_data = pd.concat([all_data, report_data], axis=1)

all_data.drop("model 1", axis = 1, inplace = True)

all_data.to_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Reports\diabetes\Report_diabetes_fairness_distcorrgan.csv", index=False)

