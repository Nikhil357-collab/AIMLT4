import matplotlib.pyplot as plt
from train import y_test, y_pred

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))

    plt.imshow(cm, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0, 1], ["Benign", "Malignant"])
    plt.yticks([0, 1], ["Benign", "Malignant"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center")

    plt.colorbar()
    plt.show()