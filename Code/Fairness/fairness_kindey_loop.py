import os
from glob import glob
import styleframe
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from fairmlhealth import report, measure

os.chdir(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis")

# Basisverzeichnis für die Datendateien
data_directory = "Data/synthetic/kidney"

num_features = ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality", "SystolicBP",
                "DiastolicBP", "FastingBloodSugar", "HbA1c", "SerumCreatinine", "BUNLevels", "GFR", "ProteinInUrine",
                "ACR", "SerumElectrolytesSodium", "SerumElectrolytesPotassium", "SerumElectrolytesCalcium",
                "SerumElectrolytesPhosphorus", "HemoglobinLevels", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
                "CholesterolTriglycerides", "NSAIDsUse", "FatigueLevels", "NauseaVomiting", "MuscleCramps", "Itching",
                "QualityOfLifeScore", "MedicalCheckupsFrequency", "MedicationAdherence", "HealthLiteracy"]

# Ausgabeverzeichnis für die Reports
report_directory = "Reports/kidney"

# Sicherstellen, dass das Ausgabeverzeichnis existiert
os.makedirs(report_directory, exist_ok=True)

# Alle CSV-Dateien im Datenverzeichnis finden
data_files = glob(os.path.join(data_directory, "*.csv"))

for file_path in data_files:
    # Dateinamen ohne Erweiterung extrahieren (für den Report-Namen)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Daten laden
    data = pd.read_csv(file_path)

    data = pd.get_dummies(data)

    scaler = MinMaxScaler()
    data[num_features] = scaler.fit_transform(data[num_features])


    # Features und Zielvariable trennen
    X = data.drop("Diagnosis", axis=1)
    y = data["Diagnosis"]

    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell trainieren (hier könnte man verschiedene Modelle ausprobieren)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Pfad für den Fairness-Report erstellen
    report_path = os.path.join(report_directory, f"{file_name}.xlsx")

    # Fairness-Report erstellen und speichern
    fairness_report = report.compare(
        test_data=X_test,
        targets=y_test,
        protected_attr=X_test["EducationLevel"],
        models=model,
        skip_performance=True
    )
    data = fairness_report.data

    data.to_excel(report_path, index=False, float_format="%.4f")

    print(f"Fairness-Report für '{file_name}' erstellt: {report_path}")
