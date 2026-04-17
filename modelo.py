from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def get_class(model_path, labels_path, image_path):
    # Load the model
    model = load_model(model_path)

    # Load the labels
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load and preprocess the image
    image = Image.open(image_path)
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)  # Resize to 224x224
    image_array = np.asarray(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = labels[predicted_index]

    return predicted_label







