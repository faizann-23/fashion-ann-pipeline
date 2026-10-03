import csv
import os

import numpy as np
import yaml
from tensorflow import keras
from tensorflow.keras import layers


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)["train"]

    keras.utils.set_random_seed(p["seed"])

    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(p["dense_units"], activation="relu"),
        layers.Dropout(p["dropout_rate"]),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")

    keys = list(hist.history.keys())
    with open("models/history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(len(hist.history[keys[0]])):
            w.writerow([i + 1] + [hist.history[k][i] for k in keys])
    print("[train] Saved models/model.h5 and models/history.csv")


if __name__ == "__main__":
    main()