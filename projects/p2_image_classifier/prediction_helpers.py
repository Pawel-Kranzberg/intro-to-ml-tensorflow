from PIL import Image
import numpy as np
import tensorflow as tf


def process_image(input_image, image_size=224):
    image = tf.cast(input_image.copy(), tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image
    

def predict(image_path, model, top_k:int):
    image = Image.open(image_path)
    image = process_image(np.asarray(image))
    image = np.expand_dims(image, axis=0)
    probabilities = model.predict(image)[0]
    top_indices = np.argpartition(probabilities, -top_k)[-top_k:]
    probabilities = probabilities[top_indices]
    top_indices += 1
    classes = top_indices.astype(str)
    return probabilities, classes
