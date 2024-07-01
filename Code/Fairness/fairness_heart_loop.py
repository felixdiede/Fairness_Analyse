import pandas as pd
import os
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from glob import glob
import styleframe

num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

os.chdir(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\synthetic\heart/heart_3")


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

print(sorted_files)

all_data = pd.DataFrame()


for file_path in sorted_files:
    # Dateinamen ohne Erweiterung extrahieren
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    print(file_name)

    # Daten laden und verarbeiten
    data = pd.read_csv(file_path)
    data["Gender"] = data["Gender"].apply(lambda x: 0 if x == "F" else 1)


    if(data["Gender"]==1).all():
        data.loc[0, "Gender"] = 0
        print("Value changed")
    elif(data["Gender"]==0).all():
        data.loc[0, "Gender"] = 1
        print("Value changed")

    data = pd.get_dummies(data)

    scaler = MinMaxScaler()
    data[num_features] = scaler.fit_transform(data[num_features])

    # Features und Zielvariable trennen
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell trainieren
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Fairness-Report erstellen
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

all_data.to_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Reports\heart/fairness_report_heart_3.csv", index=False)

all_data.drop("model 1", axis=1, inplace=True)

all_data.to_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Reports\heart/fairness_report_heart_3.csv", index=False)




