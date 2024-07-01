import pandas as pd
from fairmlhealth import report, measure
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\real\depression_original.csv")
X = data.copy()
X.drop("dataset", axis=1, inplace=True)
X["Race"] = X["Race"].apply(lambda x: 0 if x=="Black" else 1)
X = pd.get_dummies(X)
y = data["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

fairness_report = report.compare(
test_data=X_test,
targets=y_test,
protected_attr=X_test["Race"],
models=model
)
fairness_report = fairness_report.data
print(fairness_report)