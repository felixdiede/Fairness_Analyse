import pandas as pd
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\synthetic\diabetes\loop_1\diabetes_tvae_50.csv    ")

num_features = ["BMI", "MentHlth", "PhysHlth", "Age"]

data = pd.get_dummies(data)

scaler = MinMaxScaler()
data[num_features] = scaler.fit_transform(data[num_features])

X = data.drop("Diabetes_binary", axis=1)
y = data["Diabetes_binary"]

# X["Income"] = X["Income"].apply(lambda x: 0 if x < 4 else 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

fairness_report = report.compare(
    test_data = X_test,
    targets = y_test,
    protected_attr = X_test["Income"],
    models = model
)
fairness_report = fairness_report.data

print(fairness_report)

# fairness_report.to_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Reports\diabetes/fairness_diabetes_generation.csv", index=False)