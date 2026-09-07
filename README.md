# AIMLT4

# Task 4 — Classification with Logistic Regression

## Overview
Binary classification of breast-cancer observations using Logistic Regression.

**Target:** `diagnosis` — `0 = Benign`, `1 = Malignant`

## Dataset
- Rows: 569
- Predictive features: 15
- Training: 455
- Testing: 114
- Benign: 357
- Malignant: 212
- Missing values: 0

The `id` column was removed because it is an identifier, not a meaningful predictor.

## Workflow
```text
Dataset → Validation → Target Encoding → Remove ID
→ Stratified Train/Test Split → StandardScaler
→ Logistic Regression → Probabilities → Metrics
→ Threshold Analysis → Coefficient Interpretation
```

## Results at Default Threshold = 0.50

| Metric | Result |
|---|---:|
| Accuracy | **97.37%** |
| Precision | **97.56%** |
| Recall | **95.24%** |
| F1-score | **96.39%** |
| ROC-AUC | **0.9967** |

### Confusion Matrix
```text
                 Predicted
                 0      1

Actual 0        71     1
Actual 1         2    40
```

`TN=71, FP=1, FN=2, TP=40`

## Threshold Tuning
The exploratory threshold search identified:

```text
Best Threshold = 0.54
Best F1-score  = 0.9756
```

This demonstrates the precision-recall trade-off. Threshold changes do not retrain the model.

**Validation note:** The current 0.54 threshold was selected using test labels. For a production-quality workflow, select the threshold using validation data/cross-validation and keep the final test set untouched.

## Feature Interpretation
Largest absolute coefficients included:

```text
radius_worst          +2.048
texture_mean          +1.473
concave points_mean   +1.385
concavity_mean        +1.092
smoothness_worst      +0.961
symmetry_worst        +0.883
area_mean             +0.793
```

Positive coefficients increase the model's log-odds for class 1 within the fitted model; negative coefficients indicate the opposite association. Coefficients should not be interpreted as causal medical effects.

## Why Standardize?
The predictors have different numerical scales. `StandardScaler` puts them on a comparable scale and supports stable Logistic Regression optimization.

## Why Logistic Regression?
It is a strong, fast and interpretable baseline for binary classification and provides probability estimates through the sigmoid function.

## Error Handling
- Encoded `B/M` target labels as `0/1` to resolve `pos_label` errors.
- Verified actual/predicted labels.
- Corrected F1 variable overwriting from threshold iteration.
- `0.8947` is the F1 at threshold `0.80`, not the baseline F1.

## Recommended Upgrades
1. Use a `Pipeline`.
2. Add `StratifiedKFold` cross-validation.
3. Select threshold using validation/CV data.
4. Tune `C` and regularization.
5. Check multicollinearity/VIF.
6. Add ROC and Precision-Recall curves.
7. Validate on independent data.

## Example Applications
- Customer churn
- Loan default
- Fraud screening
- Employee attrition
- Spam detection
- Customer response
- Quality-control classification
- Risk classification

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib

## Project Structure
```text
task-4-logistic-regression/
├── data/
├── src/
│   ├── 01_data_loading.py
│   ├── 02_preprocessing.py
│   ├── 03_logistic_regression.py
│   ├── 04_evaluation.py
│   ├── 05_threshold_tuning.py
│   └── 06_feature_coefficients.py
├── results/
├── README.md
└── requirements.txt
```

## Conclusion
The Logistic Regression model achieved strong benchmark performance: **97.37% accuracy, 96.39% F1-score and 0.9967 ROC-AUC** at threshold 0.50. Exploratory threshold tuning found 0.54 with F1 = 0.9756. The next major improvement is validation-based threshold selection with cross-validation.

> **Disclaimer:** Educational project only. The model is not a clinical diagnostic system.
