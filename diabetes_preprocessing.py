import pandas as pd

data = pd.read_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\real\diabetes.csv")

data["Income"] = data["Income"].apply(lambda x: 0 if x < 4 else 1)

data = data.sample(int(len(data)/15))

data.to_csv(r"C:\Users\Felix\PycharmProjects\Fairness\.venv\Data_Analysis\Data\real\diabetes_generation.csv", index=False)