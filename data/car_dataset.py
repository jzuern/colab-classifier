import csv
import os
import tensorflow as tf
import sys
from PIL import Image
import numpy as np
from model.hyper_parameters import params
from random import shuffle
import scipy.io
import string

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _path_to_features(writer, label, image_path):

    print("feature label = ", label)

    pil_img = Image.open(image_path)  # open image and convert to grayscale
    res = params["architecture"]["image_resolution"]

    pil_img = pil_img.resize([res, res])

    img = np.asarray(pil_img, dtype=np.uint8)
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)

    # Create a feature
    feature = {'label': _int64_feature(label),
               'input_1': _bytes_feature(img.tostring())}

    sample = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(sample.SerializeToString())



def maybe_convert_to_tfrecords():

    root = params["paths"]["data_dir"]

    train_filename = root + 'train.tfrecords'
    eval_filename = root + 'eval.tfrecords'

    # Test if tfrecords files already exist
    if os.path.isfile(train_filename) and os.path.isfile(eval_filename):
        
        print("Training and Evaluation tfrecord files already exist!")
        return

    # get names for classes
    class_names = []

    with open(root + '/names.csv') as csvDataFile:
        csv_reader = csv.reader(csvDataFile, delimiter=';')
        for row in csv_reader:
            class_names.append(row[0])


    train_data_dir = root + 'car_data/train/'

    # open the TFRecords file
    train_writer = tf.python_io.TFRecordWriter(train_filename)

    train_counter = 0

    with open(root + '/anno_train.csv') as csvDataFile:

        csvReader = csv.reader(csvDataFile, delimiter=',')

        for row in csvReader:

            if train_counter < 100:

                print(row)
                label = int(row[5]) - 1  # compensate for class index offset of 1

                class_path = class_names[label]
                class_path = string.replace(class_path, '/', '-')

                image_path = train_data_dir + class_path + '/' + row[0]

                _path_to_features(train_writer, label, image_path)

                train_counter += 1

    eval_writer = tf.python_io.TFRecordWriter(eval_filename)

    validation_data_dir = root + 'car_data/test/'
    eval_counter = 0

    with open(root + '/anno_test.csv') as csvDataFile:

        csvReader = csv.reader(csvDataFile, delimiter=',')
        for row in csvReader:

            if eval_counter < 100:

                print(row)
                label = int(row[5]) - 1  # compensate for class index offset of 1

                class_path = class_names[label]
                class_path = string.replace(class_path, '/', '-')

                image_path = validation_data_dir + class_path + '/' + row[0]

                _path_to_features(eval_writer, label, image_path)

                eval_counter += 1

    train_writer.close()
    eval_writer.close()

    sys.stdout.flush()


def parse_fn(serialized):

    """Parse TFRecords and perform simple data augmentation."
    """

    features = \
        {
            'input_1': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['input_1']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.float32)
    print("parse_fn: image = ", image)

    res = params["architecture"]["image_resolution"]
    
    image = tf.reshape(image, [res, res, 3])

    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.

    # Randomly flip the image horizontally and vertically.
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)

    return image, label

def train_input_fn(batch_size=params["training"]["train_batch_size"], buffer_size=10000):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    tfrecords_filename = params["paths"]["data_dir"] + 'train.tfrecords'

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_filename)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.

    dataset = dataset.map(parse_fn)

    # If training then read a buffer of the given size and randomly shuffle it.
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Allow infinite reading of the data.
    num_repeat = None

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Maximum number of elements that will be buffered
    # prefetch(n) (where n is the number of elements / batches consumed by a training step)

    prefetch_buffer_size = 10

    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    print("train_input_fn: images_batch = ", images_batch)
    res = params["architecture"]["image_resolution"]

    images_batch = tf.reshape(images_batch, [-1, res, res, 3])
    print("train_input_fn: x = ", images_batch)

    x = {'input_1': images_batch}
    y = labels_batch

    return x, y


def eval_input_fn(batch_size=params["training"]["validation_batch_size"], buffer_size=10):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    tfrecords_filename = params["paths"]["data_dir"] + 'eval.tfrecords'

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_filename)

    #dataset = dataset.shuffle(buffer_size=buffer_size)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.

    dataset = dataset.map(parse_fn)

    # If testing then don't shuffle the data.
    # Only go through the data once.
    num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Maximum number of elements that will be buffered
    # prefetch(n) (where n is the number of elements / batches consumed by a training step)

    prefetch_buffer_size = 10

    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    res = params["architecture"]["image_resolution"]

    images_batch = tf.reshape(images_batch, [-1, res, res, 3])
    print("eval_input_fn: x = ", images_batch)

    x = {'input_1': images_batch}
    y = labels_batch





    return x, y


def predict_input_fn(batch_size=params["training"]["test_batch_size"], buffer_size=10):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    tfrecords_filename = params["paths"]["data_dir"] + 'eval.tfrecords'

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_filename)

    #dataset = dataset.shuffle(buffer_size=buffer_size)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.

    dataset = dataset.map(parse_fn)

    # If testing then don't shuffle the data.
    # Only go through the data once.
    num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Maximum number of elements that will be buffered
    # prefetch(n) (where n is the number of elements / batches consumed by a training step)

    prefetch_buffer_size = 10

    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    res = params["architecture"]["image_resolution"]

    images_batch = tf.reshape(images_batch, [-1, res, res, 1])
    print("eval_input_fn: x = ", images_batch)

    x = {'input_1': images_batch}
    y = labels_batch

    return x, y
