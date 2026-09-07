from feature import X, y, pd
from train import X_test, y_test, y_pred, y_prob, model, accuracy_score, StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
#from sklearn.metrics  import roc_auc_score, roc_curve
#from y_pred_prob import y_prob



precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malignant"]
))

############==================
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malignant"]
))

roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC:", round(roc_auc, 4))

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()

plt.show()

#SIGMOID FUNCTION
import numpy as np

z = np.linspace(-10, 10, 200)

sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(7, 5))

plt.plot(z, sigmoid)

plt.axhline(0.5, linestyle="--")
plt.axvline(0, linestyle="--")

plt.xlabel("Linear Score (z)")
plt.ylabel("Probability")
plt.title("Sigmoid Function")

plt.grid()
plt.show()

#THRESHOLDING
thresholds_to_test = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

for threshold in thresholds_to_test:

    pred_threshold = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred_threshold,
        zero_division=0
    )

    f1 = 2 * precision * recall / (
        precision + recall
    ) if (precision + recall) > 0 else 0

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )
    
    #COEFFICIENTS
    #===============================#=========================#
    coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print(coefficients)

from sklearn.metrics import f1_score

final_f1 = f1_score(y_test, y_pred)

print("=" * 50)
print("LOGISTIC REGRESSION FINAL RESULTS")
print("=" * 50)

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score  : {final_f1:.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
print(f"Best Threshold: {threshold:.2f}")
print(f"Best F1-score : {f1:.4f}")