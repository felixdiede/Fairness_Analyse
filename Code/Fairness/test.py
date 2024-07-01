import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from fairmlhealth import report, measure
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\real\obesity_generation.csv")

num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\synthetic\heart\heart_tvae_150.csv")
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
print(data)