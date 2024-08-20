import pandas as pd
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


num_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\real\obesity_generation.csv")

X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

scaler = MinMaxScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


model_classifier = DecisionTreeClassifier(random_state=42)
model_classifier.fit(X_train, y_train)

fairness_report = report.compare(
    test_data = X_test,
    targets = y_test,
    protected_attr = X_test["Gender"],
    models = model_classifier,
    skip_performance = True
)

fairness_report_2 = measure.bias(
    X_test[['Gender']], y_test, model_classifier.predict(X_test)
)

data = fairness_report.data

print(data)
print(fairness_report_2)