from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = Path("saved_models/asl_cnn_best.h5")
DATA_DIR = Path("data/processed/asl_alphabet")
IMG_SIZE = (64, 64)


def load_label_map():
    label_map = np.load(DATA_DIR / "label_map.npy", allow_pickle=True).item()
    idx_to_class = {v: k for k, v in label_map.items()}
    return idx_to_class


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, 64, 64, 3)
    return arr


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    idx_to_class = load_label_map()

    # change this to any image you want to test
    img_path = Path("raw/asl_alphabet_test/A_test.jpg")
    x = preprocess_image(img_path)

    preds = model.predict(x)

    # 1) main predicted class
    pred_idx = int(np.argmax(preds[0]))
    pred_class = idx_to_class[pred_idx]

    print("Image:", img_path)
    print("Predicted class:", pred_class)

    # 2) top-3 classes
    probs = preds[0]
    top_k = 3
    top_indices = probs.argsort()[-top_k:][::-1]

    print("Top 3 classes:")
    for i in top_indices:
        print(idx_to_class[i], float(probs[i]))


if __name__ == "__main__":
    main()
