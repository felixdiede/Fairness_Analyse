import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from fairmlhealth import report, measure
from sklearn.preprocessing import MinMaxScaler

num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\real/heart_generation.csv")

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

print(data)

