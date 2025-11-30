import os
import shutil
import random

SOURCE_DIR = "dataset/dataset"
TARGET_DIR = "dataset_split"
TRAIN_RATIO = 0.8

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

train_dir = os.path.join(TARGET_DIR, "train")
test_dir = os.path.join(TARGET_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_dir, class_name, img)
        )

print("Dataset successfully split into 80% training and 20% testing.")
