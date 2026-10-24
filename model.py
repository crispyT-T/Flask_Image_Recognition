"""Model loading, image preprocessing and inference.

The trained Keras model is loaded once when this module is first imported and
reused for every prediction, so the 28 MB file is read only once per process.
"""

import os

import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

# Loading model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.h5")
model = load_model(MODEL_PATH)


# Preparing and pre-processing the image
def preprocess_img(img_path):
    """Return the image as a normalised (1, 224, 224, 3) array for the model.

    Accepts anything Pillow can open: a path, a file object or a request stream.
    """
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


# Predicting function
def predict_result(predict):
    """Return the most likely digit for a preprocessed image array."""
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
