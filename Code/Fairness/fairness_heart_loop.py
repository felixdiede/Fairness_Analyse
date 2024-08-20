import pandas as pd
import os
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from glob import glob
import styleframe

num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

os.chdir(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\synthetic\heart/heart_1/loop_10")


def custom_sort(file_name):
    if file_name.endswith(".csv"):
        file_name = file_name[:-4]  # Entferne ".csv"
    parts = file_name.split("_")
    model = parts[1]
    if len(parts) == 4:
        epoch = int(parts[2])
        lambda_value = parts[3]
    else:
        epoch = int(parts[2])
        lambda_value = ""
    model_priority = {"tabfairgan": 1, "distcorrgan": 2, "multifairgan": 3, "decaf": 4, "TVAE": 5, "CTGAN": 6}
    lambda_priority = {"02": 1, "04": 2, "06": 3, "08": 4, "1": 5, "15": 6, "2": 7, "5": 8}

    return model_priority.get(model, 99), epoch, lambda_priority.get(lambda_value, 99)

file_names = os.listdir()

sorted_files = sorted(file_names, key=custom_sort)

print(sorted_files)

all_data = pd.DataFrame()


for file_path in sorted_files:
    # Dateinamen ohne Erweiterung extrahieren
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    print(file_name)

    # Daten laden und verarbeiten
    data = pd.read_csv(file_path)

    data["Gender"] = data["Gender"].apply(lambda x: 0 if x == "F" else 1)

    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    scaler = MinMaxScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    fairness_report = report.compare(
        test_data=X_test,
        targets=y_test,
        protected_attr=X_test["Gender"],
        models=model,
        skip_performance=True
    )
    data = fairness_report.data

    # Spalte 'file_name' hinzufügen
    data[file_name] = data

    all_data = pd.concat([all_data, data], axis=1)

all_data.drop("model 1", axis=1, inplace=True)

all_data.to_excel(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Reports\heart/fairness_report_heart_1_10.xlsx", index=False)




