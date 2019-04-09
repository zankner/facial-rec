import os

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

class ImageClass():
    def __init__(self,name,image_paths):
        self.name = name
        self.image_paths = image_paths
def get_dataset():
    dataset = []

    classes = os.listdir('../data/processed')
    classes.sort()
    for person in classes:
        person_path = os.path.join('../data/processed',person)
        if os.path.isdir(person_path):
            images = os.listdir(person_path)
            image_paths = [os.path.join(person_path, img) for img in images if os.path.isfile(os.path.join(person_path, img))]
            dataset.append(ImageClass(person, image_paths))
    return dataset

def filter_dataset(data, min_num_img=3):
    print(type(data))
    filtered_dataset = []
    for img in data:
        if len(img.image_paths) >= min_num_img:
            filtered_dataset.append(img)
    return filtered_dataset

data = get_dataset()
print(len(filter_dataset(data)))

