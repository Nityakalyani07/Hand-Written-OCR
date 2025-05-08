
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def run_inference(model_path, image_path):
    model = load_model(model_path)
    image = preprocess_image(image_path)
    preds = model.predict(image)
    decoded = np.argmax(preds, axis=-1)
    return decoded
