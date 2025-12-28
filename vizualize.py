import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2
import itertools
import math

def plot_history(history) -> None:
    hist = history.history

    plt.figure()
    plt.plot(hist["accuracy"], label="train acc")
    plt.plot(hist["val_accuracy"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.grid(True)

    plt.figure()
    plt.plot(hist["loss"], label="train loss")
    plt.plot(hist["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True)

    plt.show()

def plot_confusion_matrix(model, X_test, y_test, class_names) -> None:
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 8))
    disp.plot(cmap="viridis", xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix")
    plt.show()

def plot_confusion(cm: np.ndarray, class_names: list[str], title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

def show_top_cases(model, X, y, class_names, k=12) -> None:
    probs = model.predict(X, verbose=0)
    pred = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(X)), pred]
    correct = pred == y

    # Most confident wrong
    wrong_idx = np.where(~correct)[0]
    if len(wrong_idx) > 0:
        top_wrong = wrong_idx[np.argsort(conf[wrong_idx])[::-1][:k]]
        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(top_wrong, 1):
            plt.subplot(3, 4, i)
            img = (X[idx] + 1.0) * 127.5
            img = np.clip(img, 0, 255).astype(np.uint8)
            plt.imshow(img.astype(np.uint8))
            plt.axis("off")
            plt.title(f"GT:{class_names[y[idx]]}\nPR:{class_names[pred[idx]]} ({conf[idx]:.2f})")
        plt.suptitle("Most Confident Wrong")
        plt.tight_layout()
        plt.show()

    # Least confident right
    right_idx = np.where(correct)[0]
    if len(right_idx) > 0:
        low_right = right_idx[np.argsort(conf[right_idx])[:k]]
        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(low_right, 1):
            plt.subplot(3, 4, i)
            img = (X[idx] + 1.0) * 127.5
            img = np.clip(img, 0, 255).astype(np.uint8)
            plt.imshow(img.astype(np.uint8))
            plt.axis("off")
            plt.title(f"GT=PR:{class_names[y[idx]]}\nconf {conf[idx]:.2f}")
        plt.suptitle("Least Confident Correct")
        plt.tight_layout()
        plt.show()

def plot_class_report(model, X_test, y_test, class_names) -> None:
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    vals = {m: [rep[c][m] for c in class_names] for m in metrics}  # type: ignore

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, vals["precision"], width, label="precision")
    plt.bar(x,         vals["recall"],    width, label="recall")
    plt.bar(x + width, vals["f1-score"],  width, label="f1")
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, axis="y")
    plt.title("Per-class metrics (Test)")
    plt.tight_layout()
    plt.show()

def plot_conf_scatter(fruit_conf: np.ndarray, count_conf: np.ndarray, status: np.ndarray, title: str) -> None:
    plt.figure(figsize=(7, 6))

    for s, marker, label in [
        (2, "o", "both correct"),
        (1, "^", "one wrong"),
        (0, "x", "both wrong"),
    ]:
        idx = np.where(status == s)[0]
        if len(idx) == 0:
            continue
        plt.scatter(fruit_conf[idx], count_conf[idx], marker=marker, label=label)

    plt.xlabel("Fruit confidence")
    plt.ylabel("Count confidence")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    
def show_image_grid(
    image_paths: list[str],
    titles: list[str],
    grid_title: str,
    cols: int = 4,
) -> None:
    n = len(image_paths)
    rows = math.ceil(n / cols)
    
    # Dynamic sizing based on number of images
    fig_width = min(3.2 * cols, 20)
    fig_height = min(3.4 * rows, 16)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    for i, (path, title) in enumerate(zip(image_paths, titles), start=1):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(rows, cols, i)
        ax.imshow(img)
        ax.set_title(
            title,
            fontsize=9,
            fontweight="semibold",
            pad=4,
            loc="center",
            wrap=True,
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='black',
                alpha=0.7,
                edgecolor='none'
            )
        )
        ax.axis("off")
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=(0, 0, 1, 0.96 if grid_title else 1))
    plt.show()