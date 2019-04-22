import os

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split

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
            image_paths = [os.path.join(label_path, img) for img in images if os.path.isfile(os.path.join(label_path, img))]
            for img_path in image_paths:
                filenames.append(img_path)
                labels.append(label)
    filenames = np.asarray(filenames)
    labels = np.asarray(labels)
    return filenames, labels 

def final_split(input_dim,batch_size):
    data = get_files_and_labels()
    filenames = data[0]
    labels = data[1]

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(filenames, labels, test_size = 0.2)

    train_data = construct_dataset(train_inputs, train_labels, input_dim, batch_size)
    test_data = construct_dataset(test_inputs, test_labels, input_dim, 1)

    return train_data, test_data

def construct_dataset(inputs, labels, input_dim,batch_size):

    num_samples = len(inputs)

    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.shuffle(num_samples)
        dataset = dataset.map(lambda x, y:image_parse(x,y,input_dim), num_parallel_calls=3)
        dataset = dataset.map(lambda x, y:train_preprocess(x,y,batch_size), num_parallel_calls=3)
        dataset = dataset.batch(batch_size,True)
        dataset = dataset.prefetch(1)

        iterator = dataset.make_initializable_iterator()
        inputs,labels = iterator.get_next()
        init_op = iterator.initializer

    return {'inputs':inputs,'labels':labels,'init_op':init_op,'num_samples': num_samples}

def image_parse(filename, label, input_dim):
    image_string = tf.read_file(filename)

    image = tf.image.decode_jpeg(image_string,channels=3)
    
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [input_dim,input_dim])
    return image, label

def train_preprocess(image, label, batch_size):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0/255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

x, y = final_split(32, 10)
