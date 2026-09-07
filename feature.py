import pandas as pd
df = pd.read_csv("AIMLT2\\data\\breast_cancer.csv")

#Feature scaling puts numerical variables on comparable scales while avoiding test-set leakage.#
#===============================#==========================#
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})
X = df.drop(columns=["diagnosis", "id"])
y = df["diagnosis"]

print("Features:", X.shape)
print("Target:", y.shape)

print(df["diagnosis"].value_counts())
#======================================
print("\nSelected Features:")
print(X.columns.tolist())

