import os

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

class ImageClass():
    def __init__(self,name,image_paths):
        self.name = name
        self.image_paths = image_paths

def get_files_and_labels():
    filenames = [] 
    labels = [] 

    classes = os.listdir('../data/processed')
    classes.sort()
    for label in classes:
        label_path = os.path.join('../data/processed',label)
        if os.path.isdir(label_path):
            images = os.listdir(label_path)
            image_paths = [os.path.join(label_path, img) for img in images if os.path.isfile(os.path.join(person_path, img))]
            for img_path in image_paths:
                filenames.append(img_path)
                lables.append(label)
    return filenames, labels 


def construct_dataset(filenames, labels, input_dim):
    dataset = tf.data.Dataset.from_tesnor_slices((filenames, labels))
    dataset = tf.data.shuffle(len(filenames))
    dataset = tf.data.map(image_parse, num_parallel_calls=3)
    dataset = tf.data.map(train_preprocess(input_dim=input_dim), num_parallel_calls=3)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

def image_parse(filename, label, input_dim):
    image_string = tf.read_file(filename)

    image = tf.image.decode_(image_string,channels=3)
    
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [input_dim,input_dim])
    return resized_image, label

def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0/255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.clib_by_value(image, 0.0, 1.0)

    return image, label


print(get_files_and_labels()[0])
