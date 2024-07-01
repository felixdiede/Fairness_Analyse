import pandas as pd
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


num_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\real\obesity_generation.csv")

data = pd.get_dummies(data)

scaler = MinMaxScaler()
data[num_features] = scaler.fit_transform(data[num_features])

X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

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

data = fairness_report.data

print(data)