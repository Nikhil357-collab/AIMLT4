from feature import X, y
from evaluate import y_test, y_pred, y_prob
import numpy as np
from train import X_test, y_test, y_pred, y_prob, model, accuracy_score, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

threshold_values = np.arange(
    0.10,
    0.91,
    0.01
)

best_threshold = 0
best_f1 = 0

for threshold in threshold_values:

    predictions = (
        y_prob >= threshold
    ).astype(int)

    current_f1 = (
        2 *
        precision_score(
            y_test,
            predictions,
            zero_division=0
        ) *
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    ) / (
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
        +
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    ) if (
        precision_score(
            y_test,
            predictions,
            zero_division=0
        )
        +
        recall_score(
            y_test,
            predictions,
            zero_division=0
        )
    ) > 0 else 0

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold


#from sklearn.metrics import f1_score

final_f1 = f1_score(y_test, y_pred)

print("=" * 50)
print("LOGISTIC REGRESSION FINAL RESULTS")
print("=" * 50)

print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score  : {final_f1:.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
print(f"Best Threshold: {best_threshold:.2f}")
print(f"Best F1-score : {best_f1:.4f}")

print("Best Threshold:", round(best_threshold, 2))
print("Best F1-score :", round(best_f1, 4))