import pandas as pd
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\real\diabetes_generation_5000.csv")

num_features = ["BMI", "MentHlth", "PhysHlth"]

X = data.drop("Diabetes_binary", axis=1)
y = data["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

scaler = MinMaxScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# X["Income"] = X["Income"].apply(lambda x: 0 if x < 4 else 1)

model_classifier = DecisionTreeClassifier(random_state=42)
model_classifier.fit(X_train, y_train)

fairness_report = report.compare(
    test_data = X_test,
    targets = y_test,
    protected_attr = X_test["Income"],
    models = model_classifier
)
fairness_report = fairness_report.data

print(fairness_report.loc['Group Fairness', 'Equal Odds Ratio'])

print(fairness_report)

# fairness_report.to_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Reports\diabetes/fairness_diabetes_generation.csv", index=False)