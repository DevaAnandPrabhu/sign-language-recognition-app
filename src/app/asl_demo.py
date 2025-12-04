from pathlib import Path
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

# paths and constants
MODEL_PATH = Path("saved_models/asl_cnn_best.h5")
DATA_DIR = Path("data/processed/asl_alphabet")
IMG_SIZE = (64, 64)


def load_label_map():
    label_map = np.load(DATA_DIR / "label_map.npy", allow_pickle=True).item()
    idx_to_class = {v: k for k, v in label_map.items()}
    return idx_to_class


# load model + labels once
model = tf.keras.models.load_model(MODEL_PATH)
idx_to_class = load_label_map()


def classify(image: Image.Image):
    # image is a PIL image from Gradio
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    # return dict {label: probability} for Gradio Label component
    return {idx_to_class[i]: float(preds[i]) for i in range(len(preds))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload ASL hand image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted letter"),
    title="ASL Alphabet Classifier",
    description="Upload an image of an ASL hand sign to see the predicted letter.",
)

if __name__ == "__main__":
    demo.launch()
