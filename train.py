import os
import json
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

TRAIN_DIR = "dataset_split/train"
TEST_DIR = "dataset_split/test"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 25

datagen = ImageDataGenerator(rescale=1./255)

train_ds = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_ds = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

class_indices = train_ds.class_indices

with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

with open("class_labels.pkl", "wb") as f:
    pickle.dump(class_indices, f)

print("Saved class labels.")

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(class_indices), activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, epochs=EPOCHS)

print("Evaluating on test dataset...")
loss, acc = model.evaluate(test_ds)
print(f"\nTEST ACCURACY: {acc*100:.2f}%")

model.save("model.h5")
print("Model saved as model.h5")
