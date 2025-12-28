"""
Evaluation and analysis script for a two-model fruit pipeline.

This module loads two trained Keras models:
- a fruit *category* classifier (e.g., apple, banana, ...)
- a fruit *count* classifier (e.g., one, two, three, many)

It then evaluates both models on a shared test set by matching images
across folder trees using the image basename. The script reports accuracy,
classification reports, confusion matrices, confidence scatter plots, and
image grids for high-confidence mistakes and hardest examples.
"""

import os
from dataclasses import dataclass
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

import keras
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from vizualize import (plot_confusion, plot_conf_scatter,
                       show_image_grid)

FloatImageArray: TypeAlias = NDArray[np.float32]
IntLabelArray: TypeAlias = NDArray[np.int64]

# -----------------------------
# CONFIG
FRUIT_MODEL_PATH: str = "fruit_classifier.keras"
COUNT_MODEL_PATH: str = "fruit_counter.keras"

TEST_FRUIT_DIR: str = "data/fruits_category"
TEST_COUNT_DIR: str = "data/fruits_count"

IMG_SIZE: tuple[int, int] = (128, 128)

FRUIT_CLASSES: list[str] = ["apple", "banana", "cherry", "chickoo", "grapes",
                            "kiwi", "mango", "orange", "strawberry"]
COUNT_CLASSES: list[str] = ["one", "two", "three", "many"]
# -----------------------------


@dataclass(frozen=True)
class LabeledPath:
    path: str
    label: int

# -----------------------------
# Data loading
def build_index_by_basename(root_dir: str, classes: list[str]) -> dict[str, LabeledPath]:
    """
    Build an index mapping image basename to labeled file path.

    The directory layout is expected to be:
        root_dir/<class_name>/*

    Each file is indexed by its basename (filename without directories).
    This enables matching the same underlying image across multiple label
    trees (e.g., category labels and count labels) as long as basenames are
    unique and consistent.

    Parameters
    ----------
    root_dir : str
        Root directory containing one subfolder per class.
    classes : list[str]
        Ordered class folder names. The index in this list is used as the label.

    Returns
    -------
    dict[str, LabeledPath]
        Mapping from basename (e.g., "img_001.jpg") to LabeledPath(path, label).

    Raises
    ------
    FileNotFoundError
        If any expected class folder is missing.
    ValueError
        If duplicate basenames exist (ambiguous match) or no images are found.
    """
    out: dict[str, LabeledPath] = dict()

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for fname in os.listdir(class_dir):
            full = os.path.join(class_dir, fname)
            if not os.path.isfile(full):
                continue

            base = os.path.basename(full)
            if base in out:
                raise ValueError(
                    f"Duplicate basename '{base}' under {root_dir}. "
                    "Basenames must be unique to merge fruit/count correctly."
                )
            out[base] = LabeledPath(path=full, label=label_idx)

    if len(out) == 0:
        raise ValueError(f"No images found under {root_dir}")

    return out
# -----------------------------

def intersect_and_sort(
    fruit_index: dict[str, LabeledPath],
    count_index: dict[str, LabeledPath],
) -> list[str]:
    """
    Compute the set intersection of basenames across two indexes.

    This returns a *stable* sorted list of basenames that exist in both
    indexes, ensuring aligned pairing between fruit and count labels.

    Parameters
    ----------
    fruit_index : dict[str, LabeledPath]
        Basename -> labeled path index for fruit category folders.
    count_index : dict[str, LabeledPath]
        Basename -> labeled path index for fruit count folders.

    Returns
    -------
    list[str]
        Sorted basenames present in both datasets.

    Raises
    ------
    ValueError
        If there are no matching basenames between the two datasets.
    """
    common: list[str] = sorted(set(fruit_index.keys()) & set(count_index.keys()))
    if not common:
        raise ValueError("No matching basenames between fruit and count folders.")

    return common


def load_X_y(
    basenames: list[str],
    fruit_index: dict[str, LabeledPath],
    count_index: dict[str, LabeledPath],
    img_size: tuple[int, int],
) -> tuple[FloatImageArray, IntLabelArray, IntLabelArray, list[str], list[str]]:
    """
    Load aligned images and labels into memory for both tasks.

    Images are loaded using the fruit-index path (assumed to reference the
    same underlying image as the count-index path via basename), resized,
    converted to RGB float32, then preprocessed using MobileNetV2's
    `preprocess_input`.

    Parameters
    ----------
    basenames : list[str]
        Filenames used to align category and count labels.
    fruit_index : dict[str, LabeledPath]
        Basename -> labeled path index for fruit category folders.
    count_index : dict[str, LabeledPath]
        Basename -> labeled path index for fruit count folders.
    img_size : tuple[int, int]
        Desired image size as (height, width).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]
        X : np.ndarray
            Preprocessed image batch of shape (N, H, W, 3), dtype float32.
        y_fruit : np.ndarray
            Fruit category labels of shape (N,), dtype int.
        y_count : np.ndarray
            Count labels of shape (N,), dtype int.
        paths : list[str]
            Fruit image paths used for visualization (aligned with X rows).
        names : list[str]
            Basenames aligned with X rows.

    Raises
    ------
    ValueError
        If any image cannot be read.
    """
    H, W = img_size
    X: FloatImageArray = np.empty((len(basenames), H, W, 3), dtype=np.float32)
    y_fruit: IntLabelArray = np.empty((len(basenames),), dtype=np.int64)
    y_count: IntLabelArray = np.empty((len(basenames),), dtype=np.int64)
    paths: list[str] = []

    i: int; base: str
    for i, base in enumerate(basenames):
        fruit_lp = fruit_index[base]
        count_lp = count_index[base]

        img = cv2.imread(fruit_lp.path)
        if img is None:
            raise ValueError(f"Failed to read image: {fruit_lp.path}")

        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        X[i] = img
        y_fruit[i] = fruit_lp.label
        y_count[i] = count_lp.label
        paths.append(fruit_lp.path)

    X = np.array(preprocess_input(X, data_format="channels_last"), dtype=np.float32)

    return X, y_fruit, y_count, paths, basenames

# -----------------------------
# Main pipeline
def main() -> None:
    """
    Run evaluation for both fruit category and fruit count models.

    The function:
    1. Loads both trained models.
    2. Builds basename-based indexes for category and count folders.
    3. Intersects basenames to create an aligned evaluation set.
    4. Loads images into memory and preprocesses them.
    5. Runs predictions for both models.
    6. Computes accuracy metrics, reports, and confusion matrices.
    7. Visualizes confidence relationships and example grids:
       - high-confidence mistakes (wrong on either task)
       - hardest images (lowest combined confidence)

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If models cannot be loaded or required data is missing.
    """
    fruit_model = keras.models.load_model(FRUIT_MODEL_PATH)
    count_model = keras.models.load_model(COUNT_MODEL_PATH)

    if fruit_model is None or count_model is None:
        raise ValueError("Models not found")

    fruit_index: dict[str, LabeledPath] = build_index_by_basename(TEST_FRUIT_DIR, FRUIT_CLASSES)
    count_index: dict[str, LabeledPath] = build_index_by_basename(TEST_COUNT_DIR, COUNT_CLASSES)

    basenames: list[str] = intersect_and_sort(fruit_index, count_index)

    X, y_fruit, y_count, fruit_paths, names = load_X_y(
        basenames=basenames,
        fruit_index=fruit_index,
        count_index=count_index,
        img_size=IMG_SIZE,
    )

    fruit_probs: FloatImageArray = fruit_model.predict(X, verbose=1)
    count_probs: FloatImageArray = count_model.predict(X, verbose=1)

    fruit_pred: FloatImageArray = np.argmax(fruit_probs, axis=1)
    count_pred: FloatImageArray = np.argmax(count_probs, axis=1)

    fruit_conf: FloatImageArray = np.max(fruit_probs, axis=1)
    count_conf: FloatImageArray = np.max(count_probs, axis=1)

    combined_pred_str: list[str] = [f"{FRUIT_CLASSES[f]} + {COUNT_CLASSES[c]}" for f, c in zip(fruit_pred, count_pred)]
    combined_true_str: list[str] = [f"{FRUIT_CLASSES[f]} + {COUNT_CLASSES[c]}" for f, c in zip(y_fruit, y_count)]

    fruit_acc: float = float(accuracy_score(y_fruit, fruit_pred))
    count_acc: float = float(accuracy_score(y_count, count_pred))
    both_correct: float = np.mean((y_fruit == fruit_pred) & (y_count == count_pred))

    print("\n=== Accuracy ===")
    print(f"Fruit accuracy:  {fruit_acc:.4f}")
    print(f"Count accuracy:  {count_acc:.4f}")
    print(f"Both correct:    {both_correct:.4f}")

    print("\n=== Fruit classification report ===")
    print(classification_report(y_fruit, fruit_pred, target_names=FRUIT_CLASSES))

    print("\n=== Count classification report ===")
    print(classification_report(y_count, count_pred, target_names=COUNT_CLASSES))

    cm_fruit = confusion_matrix(y_fruit, fruit_pred)
    cm_count = confusion_matrix(y_count, count_pred)
    plot_confusion(cm_fruit, FRUIT_CLASSES, "Fruit confusion matrix")
    plot_confusion(cm_count, COUNT_CLASSES, "Count confusion matrix")

    status: NDArray[np.int64] = np.zeros(len(names), dtype=int)
    only_one_correct: NDArray = ((y_fruit == fruit_pred) ^ (y_count == count_pred))
    both: NDArray = ((y_fruit == fruit_pred) & (y_count == count_pred))
    status[only_one_correct] = 1
    status[both] = 2

    plot_conf_scatter(fruit_conf, count_conf, status, "Fruit vs Count confidence")

    # Sample
    print("\n=== Sample predictions ===")
    i: int
    for i in np.random.choice(range(len(names)), size=15):
        print(f"{names[i]:30s}  TRUE: {combined_true_str[i]:25s}  PRED: {combined_pred_str[i]:25s}")

    wrong_idx: NDArray[np.int64] = np.where(status != 2)[0]
    if len(wrong_idx) > 0:
        combined_conf = fruit_conf * count_conf
        wrong_sorted = wrong_idx[np.argsort(-combined_conf[wrong_idx])]

        top_k: int = min(12, len(wrong_sorted))
        pick = wrong_sorted[:top_k]

        img_paths: list[str] = []
        titles: list[str] = []
        idx: int
        for idx in pick:
            img_paths.append(fruit_paths[idx])
            titles.append(
                    f"True:  {combined_true_str[idx]}\n"
                    f"Pred:  {combined_pred_str[idx]}\n"
                    f"Fruit conf: {fruit_conf[idx]:.2f}\n"
                    f"Count conf: {count_conf[idx]:.2f}")

        show_image_grid(img_paths, titles, f"Top {top_k} high-confidence mistakes (combined)", cols=4)
# -----------------------------

if __name__ == "__main__":
    main()
