"""
Fruit count classification training script using MobileNetV2.

This module trains a transfer-learning model to classify images into
four count-based categories (one, two, three, many). It includes data
loading, preprocessing, augmentation, model training, evaluation, and
visual reporting.

The final trained model and architecture diagram are saved to disk.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import TypeAlias

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from keras.callbacks import History, EarlyStopping
import keras

from utils import show_data, load_data, show_augmented_batch
from vizualize import (plot_history, plot_confusion_matrix,
                       plot_class_report, show_top_cases)

FloatImageArray: TypeAlias = NDArray[np.float32]
IntLabelArray: TypeAlias = NDArray[np.int64]
KerasIterator: TypeAlias = keras.utils.Sequence

keras.utils.set_random_seed(42)

def train_model(
    train_gen: KerasIterator,
    val_gen: KerasIterator,
    steps_per_epoch: int,
    validation_steps: int,
    epochs: int = 10,
) -> tuple[keras.Model, History]:
    """
    Build, compile, and train a MobileNetV2-based classifier.

    This function applies transfer learning using a pretrained
    MobileNetV2 backbone with frozen weights. A custom classification
    head is added to predict one of four count-based classes.

    Model architecture:
    - MobileNetV2 (feature extractor, frozen)
    - GlobalAveragePooling2D
    - Dense layer with L1/L2 regularization
    - Dropout for regularization
    - Softmax output layer (4 classes)

    Early stopping monitors validation accuracy to prevent overfitting.

    Parameters
    ----------
    train_gen : KerasIterator
        Generator yielding augmented training images and labels.
    val_gen : KerasIterator
        Generator yielding validation images and labels.
    steps_per_epoch : int
        Number of training steps per epoch.
    validation_steps : int
        Number of validation steps per epoch.
    epochs : int, optional
        Maximum number of training epochs, by default 10.

    Returns
    -------
    tuple[keras.Model, History]
        The trained Keras model and the training history object.
    """
    base_model: keras.Model = keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False)
    base_model.trainable = False

    model: keras.Model = Sequential(
        [
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            Dense(128, activation="tanh", kernel_regularizer=keras.regularizers.l1_l2(1e-3, 1e-3)),
            Dropout(0.3),
            Dense(4, activation="softmax"),
        ]
    )

    early_stop: EarlyStopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    mode="max",
    start_from_epoch=20,
    verbose=1)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3), #type: ignore
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    history: History = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[early_stop]
    )

    return model, history

def main() -> None:
    """
    Run the full training and evaluation workflow for fruit count prediction.

    Workflow steps:
    1. Load image data labeled by object count.
    2. Shuffle and visualize samples.
    3. Preprocess images for MobileNetV2.
    4. Split data into training, validation, and test sets.
    5. Apply data augmentation to the training set.
    6. Train the transfer-learning model.
    7. Evaluate performance on the test set.
    8. Save the trained model and generate diagnostic plots.

    Returns
    -------
    None
    """
    images, labels, names = load_data(data_type="Count")

    rng: Generator = np.random.default_rng(seed=42)
    indices: NDArray[np.int64] = rng.permutation(len(images))

    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]
    names = [names[i] for i in indices]

    show_data(images=images, names=names, max_images=12)

    X: FloatImageArray = np.array(preprocess_input(np.array(images, dtype=np.float32), data_format="channels_last"), dtype=np.float32)
    y: IntLabelArray = np.array(labels, dtype=np.int64)

    X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val \
        = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

    train_datagen: ImageDataGenerator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen: ImageDataGenerator = ImageDataGenerator()

    batch_size: int = 32

    train_gen: KerasIterator = train_datagen.flow(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    val_gen: KerasIterator = val_datagen.flow(
        X_val, y_val, batch_size=batch_size, shuffle=False
    )

    class_names: list[str] = ["one", "two", "three", "many"]
    show_augmented_batch(train_gen=train_gen, class_names=class_names, max_images=16, assume_bgr=False)

    steps_per_epoch: int = int(np.ceil(len(X_train) / batch_size))
    validation_steps: int = int(np.ceil(len(X_val) / batch_size))

    model, history = train_model(
        train_gen=train_gen,
        val_gen=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=100,
    )

    _, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    model.save("fruit_counter.keras")

    plot_history(history=history)
    plot_confusion_matrix(model=model, X_test=X_test, y_test=y_test, class_names=class_names)
    plot_class_report(model=model, X_test=X_test, y_test=y_test, class_names=class_names)
    show_top_cases(model=model, X=X_test, y=y_test, class_names=class_names, k=12)
    plot_model(model=model, to_file="model_counter.png")

if __name__ == "__main__":
    main()
