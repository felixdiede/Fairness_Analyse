import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from fairmlhealth import report, measure



data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\real\kidney.csv")

X = data.copy()

X.drop(["PatientID", "DoctorInCharge", "Diagnosis"], inplace=True, axis=1)

y = data["Diagnosis"]

columns_temp = X.columns

# X["Ethnicity"] = X["Ethnicity"].apply(lambda x: 0 if x==2 else 1)
X["EducationLevel"] = X["EducationLevel"].apply(lambda x: 0 if x==3 else 1)
# X["SocioeconomicStatus"] = X["SocioeconomicStatus"].apply(lambda x: 0 if x==2 else 1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns = columns_temp)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

fairness_report = report.compare(
    test_data=X_test,
    targets=y_test,
    protected_attr=X_test["EducationLevel"],
    models=model,
    skip_performance=True
)

result = fairness_report.data
print(result)

fairness_report.to_html("fairness_kidney.html")