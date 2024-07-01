import pandas as pd
import os
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from glob import glob
import styleframe

num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\synthetic/heart/heart_2/heart_tvae_100.csv")

data["Gender"] = data["Gender"].apply(lambda x: 0 if x == "F" else 1)

data = pd.get_dummies(data)

scaler = MinMaxScaler()
data[num_features] = scaler.fit_transform(data[num_features])

X = data.drop("HeartDisease", axis = 1)
y = data["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

fairness_report = report.compare(
    test_data = X_test,
    targets = y_test,
    protected_attr = X_test["Gender"],
    models = model,
    skip_performance = True
)

results = fairness_report.data

print(results)