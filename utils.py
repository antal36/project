"""
Utility functions for dataset inspection, loading, visualization,
and augmentation preview for fruit image classification tasks.

This module provides helpers for:
- Inspecting raw image sizes on disk
- Loading and preprocessing image datasets
- Visualizing sample images and augmented batches
"""

import os
import cv2
from pathlib import Path
from cv2.typing import MatLike
import numpy as np
import keras
from typing import Sequence, Literal

def print_sizes(root_dir: str = "data"):
    """
    Print the width and height of all JPG images in a directory tree.

    This function recursively walks through the given directory and
    prints image dimensions for each `.jpg` file found. It is useful
    for inspecting dataset consistency and identifying problematic files.

    Parameters
    ----------
    root_dir : str, optional
        Root directory to search for images, by default "data".

    Returns
    -------
    None
    """
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue

                height, width, _ = img.shape
                print(f"{img_path}: {width} x {height}")

import matplotlib.pyplot as plt
import math

def show_data(images: list[MatLike], names: Sequence[str], max_images: int = 16) -> None:
    """
    Display a grid of sample images with their corresponding labels.

    Parameters
    ----------
    images : list[MatLike]
        List of image arrays (RGB expected for correct display).
    names : Sequence[str]
        Class names corresponding to each image.
    max_images : int, optional
        Maximum number of images to display, by default 16.

    Returns
    -------
    None
    """
    num_images = min(len(images), max_images)
    cols = 4
    rows = math.ceil(num_images / cols)

    plt.figure(figsize=(12, 3 * rows))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.title(names[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def load_data(data_type: Literal["Count", "Category"], data_dir: str | Path = "data/",
              img_size: int = 128) -> \
    tuple[list[MatLike], list[int], list[str]]:
    """
    Load and preprocess fruit image datasets from disk.

    The dataset structure is expected to follow:

    - Category classification:
      data/fruits_category/<class_name>/*.jpg
    - Count classification:
      data/fruits_count/<class_name>/*.jpg

    Images are:
    - Loaded with OpenCV
    - Converted from BGR to RGB
    - Filtered by minimum size
    - Resized to a fixed square resolution

    Parameters
    ----------
    data_type : {"Count", "Category"}
        Type of classification task.
    data_dir : str | Path, optional
        Root data directory, by default "data/".
    img_size : int, optional
        Target image height and width in pixels, by default 128.

    Returns
    -------
    tuple[list[MatLike], list[int], list[str]]
        - images: list of resized RGB images
        - labels: integer class labels
        - names: class name for each image
    """

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
        
    images: list[MatLike] = []
    labels: list[int] = []
    names: list[str] = []

    fruit_label: dict[str, int]
    if data_type == "Category":
        fruit_label = dict(zip(["apple", "banana", "cherry", "chickoo", "grapes",
                                   "kiwi", "mango",  "orange", "strawberry"], range(9)))
        data_dir = data_dir / "fruits_category"
    else:
        fruit_label = {"one": 0, "two": 1, "three": 2, "many": 3}
        data_dir = data_dir / "fruits_count"

    name: str; label: int
    for name, label in fruit_label.items():

        path: Path = data_dir / f"{name}"

        fruit_files: list[Path] = list(path.glob("*"))

        img_path: Path
        for img_path in fruit_files:
            img: MatLike | None = cv2.imread(str(img_path))

            if img is None:
                print(f"couldnt load {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.shape[0] >= img_size and img.shape[1] >= img_size:
                resized_img: MatLike = cv2.resize(img, (img_size, img_size))

                images.append(resized_img)
                labels.append(label)
                names.append(name)
    return (images, labels, names)


def show_augmented_batch(
    train_gen: keras.utils.Sequence,
    class_names: Sequence[str],
    max_images: int = 16,
    cols: int = 4,
    assume_bgr: bool = True,
) -> None:
    """
    Visualize a single augmented batch from a Keras data generator.

    This function retrieves one batch from a Keras iterator and displays
    augmented images in a grid along with their class labels.

    It supports both sparse labels and one-hot encoded labels.

    Parameters
    ----------
    train_gen : keras.utils.Sequence
        Iterator returned by ImageDataGenerator.flow(...).
    class_names : Sequence[str]
        Mapping from class index to human-readable label.
    max_images : int, optional
        Maximum number of images to display, by default 16.
    cols : int, optional
        Number of columns in the display grid, by default 4.
    assume_bgr : bool, optional
        If True, images are assumed to be in BGR format (OpenCV style)
        and will be converted to RGB for display.

    Returns
    -------
    None
    """
    batch = next(iter(train_gen))
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x_batch, y_batch = batch[0], batch[1]
    else:
        raise ValueError("train_gen did not yield (images, labels).")

    x_batch = np.asarray(x_batch)
    y_batch = np.asarray(y_batch)

    if y_batch.ndim > 1 and y_batch.shape[-1] > 1:
        y_idx = np.argmax(y_batch, axis=-1)
    else:
        y_idx = y_batch.astype(int).reshape(-1)

    n = min(int(x_batch.shape[0]), max_images)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(12, 3 * rows))

    for i in range(n):
        img = x_batch[i]
        img = (img + 1.0) * 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)

        if img.dtype != np.uint8:
            img_disp = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_disp = img

        if assume_bgr and img_disp.shape[-1] == 3:
            img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)

        label_i = int(y_idx[i])
        title = class_names[label_i] if 0 <= label_i < len(class_names) else str(label_i)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_disp)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
