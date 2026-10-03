import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras


def main():
    model = keras.models.load_model("models/model.h5", compile=False)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    preds = np.argmax(model.predict(x_test, verbose=0), axis=1)

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
    fig.savefig("models/confusion_matrix.png", dpi=120)

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)
    print(f"[evaluate] test_loss={loss:.4f} test_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()