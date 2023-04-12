# -*- coding: utf-8 -*-
"""A pure script for identifying flower species in flower images,
using a previously trained TensorFlow deep learning model.
Prints out the flower class predictions.
"""


import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

import prediction_helpers


# Parsing of command line arguments:
CLA_parser = argparse.ArgumentParser(
	description='Predict flower classes in an image.'
)
CLA_parser.add_argument(
	'path_to_image',
	type=str,
	nargs=1,
    help='path and name of the image'
)
CLA_parser.add_argument(
	'path_to_model',
	type=str,
	nargs=1,
    help='path and name of the classification model'
)
CLA_parser.add_argument(
	'--top_k',
    metavar='top_k',
	type=int,
	nargs='?',
    default=1,
    help='number of the top k most likely classes to return'
)
CLA_parser.add_argument(
	'--category_names',
    metavar='category_names',
	type=str,
	nargs='?',
    help='path and name of a JSON file mapping labels to flower names'
)

arguments = CLA_parser.parse_args()


# Loading the model:
try:
    model = tf.keras.models.load_model(
        filepath=arguments.path_to_model[0]
    )
except:
    model = tf.keras.models.load_model(
        filepath=arguments.path_to_model[0],
        compile=False,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
# print(model.summary())


# Class prediction:
probabilities, classes = prediction_helpers.predict(
    arguments.path_to_image[0],
    model,
    arguments.top_k
)


# Loading class labels:
if arguments.category_names is not None:
    with open(arguments.category_names, 'r') as f:
        class_names = json.load(f)
    classes = [class_names[k] for k in classes]


# Print results:
print(f'\nTop {arguments.top_k} classes & probabilities:\n')
for c, p in zip(classes, probabilities):
    print(f'{c}: {p:.3%}')
print('\n')
