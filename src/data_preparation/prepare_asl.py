from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# Raw ASL alphabet images (A, B, ..., space, nothing, del)
DATA_ROOT = Path("raw/asl_alphabet_train")

# Where to save processed numpy arrays
OUTPUT_DIR = Path("data/processed/asl_alphabet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (64, 64)
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42



def load_images():
    class_names = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    X = []
    y = []

    max_per_class = 1000  # limit to make preprocessing faster

    for cls in class_names:
        cls_dir = DATA_ROOT / cls
        images_added = 0
        for fname in os.listdir(cls_dir):
            if images_added >= max_per_class:
                break
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = cls_dir / fname
            img = Image.open(path).convert("RGB")
            img = img.resize(IMG_SIZE)
            X.append(np.array(img, dtype="float32") / 255.0)
            y.append(class_to_idx[cls])
            images_added += 1

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int64")
    return X, y, class_to_idx


def main():
    X, y, class_to_idx = load_images()
    print("Loaded", len(X), "images")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=1 - val_ratio,
        random_state=RANDOM_STATE,
        stratify=y_tmp,
    )

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    np.save(OUTPUT_DIR / "label_map.npy", class_to_idx)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
