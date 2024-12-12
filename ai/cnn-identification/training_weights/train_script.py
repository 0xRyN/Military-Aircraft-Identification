# %% [markdown]
# ## Extract, Transform and Load (ETL) Step
#
# We are going to use a custom CNN for object identification.

# %% [markdown]
# ### Setup

import json
import math

# %%
import os
import random
import shutil
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras.applications import EfficientNetB3
from sklearn.utils import class_weight
from tqdm import tqdm

ROOT_DIR = "datasets/aircraft"
DATASET_DIR = "../dataset/crop"

# %% [markdown]
# ## Classes Mapping
#
# We also need to provide a classes mapping, here is how we do that

# %%
CLASSES_RAW = [
    "a10",
    "a400m",
    "ag600",
    "ah64",
    "av8b",
    "an124",
    "an22",
    "an225",
    "an72",
    "b1",
    "b2",
    "b21",
    "b52",
    "be200",
    "c130",
    "c17",
    "c2",
    "c390",
    "c5",
    "ch47",
    "cl415",
    "e2",
    "e7",
    "ef2000",
    "f117",
    "f14",
    "f15",
    "f16",
    "f22",
    "f35",
    "f4",
    # IN DATA AS F18
    "f18",
    "h6",
    "j10",
    "j20",
    "jas39",
    "jf17",
    "jh7",
    "kc135",
    "kf21",
    "kj600",
    "ka27",
    "ka52",
    "mq9",
    "mi24",
    "mi26",
    "mi28",
    "mig29",
    "mig31",
    "mirage2000",
    "p3",
    "rq4",
    "rafale",
    "sr71",
    "su24",
    "su25",
    "su34",
    "su57",
    "tb001",
    "tb2",
    "tornado",
    "tu160",
    "tu22m",
    "tu95",
    "u2",
    "uh60",
    "us2",
    "v22",
    "vulcan",
    "wz7",
    "xb70",
    "y20",
    "yf23",
    "z19",
]

CLASSES = {model: i for i, model in enumerate(CLASSES_RAW)}


def get_class_id(class_str: str):
    return CLASSES[class_str.lower()]


print(CLASSES)

# %% [markdown]
# ### Defining Dataset Constants

# %%
IMG_WIDTH = 256  # Keras default
IMG_HEIGHT = 256  # Keras default
IMG_CHANNELS = 3  # RGB

BATCH_SIZE = 32
EPOCHS = 10

# %% [markdown]
# ### Extracting the Dataset
#
# We need to first extract the x and y data from the dataset (image paths and labels).
#
# ### Splitting the Dataset
#
# We need to split out dataset into three parts, train, validation and test.
#
# We're going to use (80%, 10%, 10%) repartition for now.


# %%
def split_dataset(
    paths: list[str], classes: list[int], seed: int = None
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    if not seed:
        seed = time.time()

    random.seed(seed)

    for i in tqdm(range(len(paths))):
        rand = random.randint(1, 10)

        # Validation (10%)
        if rand == 9:
            x_val.append(paths[i])
            y_val.append(classes[i])

        # Test (10%)
        elif rand == 10:
            x_test.append(paths[i])
            y_test.append(classes[i])

        # Train (80%)
        else:
            x_train.append(paths[i])
            y_train.append(classes[i])

    return x_train, y_train, x_val, y_val, x_test, y_test


def extract_dataset(dataset_dir: str = DATASET_DIR, seed: int = None) -> None:
    aircraft_filepaths = []
    aircraft_classes = []
    for aircraft_dir in tqdm(os.listdir(dataset_dir)):
        aircraft_class = get_class_id(aircraft_dir)

        dir_path = os.path.join(dataset_dir, aircraft_dir)

        for aircraft_img in os.listdir(dir_path):
            aircraft_img_path = os.path.join(dir_path, aircraft_img)

            aircraft_filepaths.append(aircraft_img_path)
            aircraft_classes.append(aircraft_class)

    print(f"Found {len(aircraft_filepaths)} aircraft images")

    return aircraft_filepaths, aircraft_classes


paths, classes = extract_dataset()

x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(paths, classes)

print(f"Train: {len(x_train)}")
print(f"Validation: {len(x_val)}")
print(f"Test: {len(x_test)}")

# %% [markdown]
# ### Transforming the Dataset
#
# We need to transform the dataset into a format that can be used by the model.
#
# This includes
#
# - Loading the images as tensors
# - Resizing the images
# - Normalizing the images (done later by the model)
#
# ### Lazy Loading and Tensorflow Datasets
#
# We're going to use the `tf.data.Dataset` API to load the images lazily.
#
# This is done to avoid loading all the images into RAM at once (10GB+ of images).
#
# We also need to shuffle the dataset to avoid biasing the model since the images are ordered by class.
#
# Notes:
#
# - We're going to use a batch size of 32 for now
# - We're going to use a prefetch buffer, automatically tuned by Tensorflow
#


# %%
class AircraftDataGenerator:
    def __init__(
        self, filepaths: list[str], classes: list[int], batch_size: int = BATCH_SIZE
    ):
        self.filepaths = filepaths
        self.classes = classes
        self.batch_size = batch_size

    def load_image(self, filepath: str) -> tf.Tensor:
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
        return image

    def create_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((self.filepaths, self.classes))

        dataset = dataset.shuffle(
            buffer_size=len(self.filepaths), reshuffle_each_iteration=True
        )

        dataset = dataset.map(
            lambda x, y: (self.load_image(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.batch(self.batch_size)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


train_generator = AircraftDataGenerator(x_train, y_train)
val_generator = AircraftDataGenerator(x_val, y_val)
test_generator = AircraftDataGenerator(x_test, y_test)

train_dataset = train_generator.create_dataset()
val_dataset = val_generator.create_dataset()
test_dataset = test_generator.create_dataset()

# %% [markdown]
# ### Building the Model Tests
#
#


# %%
def build_data_augmentation():
    """Creates a data augmentation sequential model."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
        ],
        name="data_augmentation",
    )


def build_base_model(include_data_augmentation=False):
    """Builds the base model with or without data augmentation."""
    layers = []

    if include_data_augmentation:
        data_augmentation = build_data_augmentation()
        layers.append(data_augmentation)

    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="max",
    )
    layers.append(base_model)
    layers.append(keras.layers.BatchNormalization(name="batch_normalization"))
    layers.append(
        keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.01),
            name="dense_256",
        )
    )
    layers.append(keras.layers.Dropout(0.2, name="dropout_0.2"))
    layers.append(
        keras.layers.Dense(len(CLASSES), activation="softmax", name="output_layer")
    )

    model = keras.Sequential(layers, name="efficientnetb3_model")
    return model


def compile_model(model, optimizer="adam", learning_rate=0.001):
    """Compiles the model with specified optimizer and learning rate."""
    if optimizer.lower() == "adamax":
        optimizer_instance = keras.optimizers.Adamax(learning_rate=learning_rate)
    else:
        optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer_instance,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_class_weights(y_train):
    """Calculates class weights for handling class imbalance."""
    y_train_np = np.array(y_train)
    class_weights_values = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train_np), y=y_train_np
    )
    class_weights_dict = dict(enumerate(class_weights_values))
    return class_weights_dict


def get_callbacks():
    """Defines callbacks for training."""

    def lr_decay(epoch):
        return 0.001 * math.pow(0.666, epoch)

    lr_decay_callback = keras.callbacks.LearningRateScheduler(lr_decay, verbose=1)
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="best_model_sofar.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    return [lr_decay_callback, early_stop_callback, checkpoint_callback]


# %% [markdown]
# ### Test Multiple Configurations

# %%


def train_basic():
    """Trains the basic model without any enhancements."""
    print("Training Basic Model...")
    model = build_base_model(include_data_augmentation=False)
    model = compile_model(model, optimizer="adam", learning_rate=0.001)
    model.summary()

    history = model.fit(
        train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1
    )

    model.evaluate(test_dataset)
    model.save("basic_model.keras")
    print("Basic Model saved")
    return history


def train_adamax():
    """Trains the model using the Adamax optimizer."""
    print("Training with Adamax Optimizer...")
    model = build_base_model(include_data_augmentation=False)
    model = compile_model(model, optimizer="adamax", learning_rate=0.001)
    model.summary()

    history = model.fit(
        train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1
    )

    model.evaluate(test_dataset)
    model.save("adamax_model.keras")
    print("Adamax Model saved")
    return history


def train_data_augmentation():
    """Trains the model with data augmentation."""
    print("Training with Data Augmentation...")
    model = build_base_model(include_data_augmentation=True)
    model = compile_model(model, optimizer="adam", learning_rate=0.001)
    model.summary()

    history = model.fit(
        train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1
    )

    model.evaluate(test_dataset)
    model.save("data_augmentation_model.keras")
    print("Data Augmentation Model saved")
    return history


def train_class_weights():
    """Trains the model with class weights to handle class imbalance."""
    print("Training with Class Weights...")
    model = build_base_model(include_data_augmentation=False)
    model = compile_model(model, optimizer="adam", learning_rate=0.001)
    model.summary()

    class_weights_dict = get_class_weights(y_train)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        verbose=1,
    )

    model.evaluate(test_dataset)
    model.save("class_weights_model.keras")
    print("Class Weights Model saved")
    return history


def train_full_no_cb():
    """Trains the full model with all enhancements and callbacks."""
    print("Training Full Model with All Enhancements (No CallBacks)...")
    model = build_base_model(include_data_augmentation=True)
    model = compile_model(model, optimizer="adamax", learning_rate=0.001)
    model.summary()

    class_weights_dict = get_class_weights(y_train)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        verbose=1,
    )

    model.evaluate(test_dataset)
    model.save("full_model_no_cb.keras")
    print("Full Model No Callbacks saved")
    return history


def train_full_cb():
    """Trains the full model with all enhancements and callbacks."""
    print("Training Full Model with All Enhancements (With CallBacks)...")
    model = build_base_model(include_data_augmentation=True)
    model = compile_model(model, optimizer="adamax", learning_rate=0.001)
    model.summary()

    class_weights_dict = get_class_weights(y_train)
    callbacks = get_callbacks()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1,
    )

    model.evaluate(test_dataset)
    model.save("full_model_cb.keras")
    print("Full Model Callbacks saved")
    return history


def train_optimal():
    """Trains the full model with all enhancements and callbacks."""
    print("Training Full Model Optimally (30 EPOCHS)")
    model = build_base_model(include_data_augmentation=True)
    model = compile_model(model, optimizer="adamax", learning_rate=0.001)
    model.summary()

    callbacks = get_callbacks()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=callbacks,
        verbose=1,
    )

    model.evaluate(test_dataset)
    model.save("full_model_optimal.keras")
    print("Full Model Optimal saved")
    return history


# %% [markdown]
# ### And... Benchmark the Models


# %%
def main():
    """Main function that runs all training configurations and logs their performance."""
    training_functions = [train_optimal]

    results = []

    for train_fn in training_functions:
        result = train_fn()
        result.history["Training Name"] = train_fn.__name__
        results.append(result)

    with open("training_histories.json", "w") as f:
        json.dump([history.history for history in results], f)


# %% [markdown]
# ### Running the Code and Training the Models

# %%
main()
