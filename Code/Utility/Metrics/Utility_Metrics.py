import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def trtr(real_data, target):
    X = real_data.drop(target, axis=1)
    y = real_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    classifiers = [
        RandomForestClassifier(n_estimators=100, n_jobs=3, random_state=9),
        KNeighborsClassifier(n_neighbors=10, n_jobs=3),
        DecisionTreeClassifier(random_state=9),
        SVC(C=100, max_iter=300, kernel="linear", probability=True, random_state=9),
        MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=9)
    ]

    results = {}
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[clf.__class__.__name__] = classification_report(y_test, y_pred, output_dict=True)

    results_df = pd.DataFrame(results).transpose().drop(columns=["macro avg", "weighted avg"])
    print(results_df.to_markdown(numalign="left", stralign="left", floatfmt=".4f"))


def tstr(real_data, synthetic_data, target):
    real_data = pd.get_dummies(real_data)
    synthetic_data = pd.get_dummies(synthetic_data)

    Xs = synthetic_data.drop(target, axis=1)
    ys = synthetic_data[target]

    Xr = synthetic_data.drop(target, axis=1)
    yr = synthetic_data[target]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(Xs, ys, test_size=0.2, random_state=0)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(Xr, yr, test_size=0.2, random_state=0)

    classifiers = [
        RandomForestClassifier(n_estimators=100, n_jobs=3, random_state=9),
        KNeighborsClassifier(n_neighbors=10, n_jobs=3),
        DecisionTreeClassifier(random_state=9),
        SVC(C=100, max_iter=300, kernel="linear", probability=True, random_state=9),
        MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=9)
    ]

    results = {}
    for clf in classifiers:
        clf.fit(X_train_s, y_train_s)
        y_pred = clf.predict(X_test_r)
        results[clf.__class__.__name__] = classification_report(y_test_r, y_pred, output_dict=True)

    results_df = pd.DataFrame(results).transpose()
    print(results_df.to_markdown(numalign="left", stralign="left", floatfmt=".4f"))



