from keras.preprocessing import image
from densenet import DenseNetImageNet121, preprocess_input, decode_predictions
import numpy as np

SIZE = 224

# Run once when the artifact is built to download weights and label mappings
def build():
    # Downloads label mappings on first run
    decode_predictions(np.empty([1, 1000]))
    # Downloads weights on first run
    setup()

# Run once when model is booted up so multiple inferrences can be run efficiently
def setup():
    return DenseNetImageNet121(input_shape=(SIZE, SIZE, 3))

# Run a single inferrence on the model.
# The first argument is the return value of setup(), the second is the path to image
# to run inferrence on.
def infer(model, image_path):
    img = image.load_img(image_path, target_size=(SIZE, SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    preds_with_labels = decode_predictions(preds)
    return dict((t[1], t[2]) for t in preds_with_labels[0])
