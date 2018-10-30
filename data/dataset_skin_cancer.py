import csv
import os
import tensorflow as tf
import sys
from PIL import Image
import numpy as np
from model.hyper_parameters import params
from random import shuffle

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def maybe_convert_to_tfrecords():

    root = params["paths"]["data_dir"]

    train_filename = root + 'train.tfrecords'
    eval_filename = root + 'eval.tfrecords'
    csv_file = root + 'HAM10000_metadata.csv'

    # Test if tfrecords files already exist
    if os.path.isfile(train_filename) and os.path.isfile(eval_filename):
        print("Training and Evaluation tfrecord files already exist!")
        return

    label_dict = {"bkl": 0,
                  "nv": 1,
                  "mel": 2,
                  "bcc": 3,
                  "akiec": 4,
                  "vasc": 5,
                  "df": 6}


    lesion_ids = []
    image_paths = []
    dxs = []

    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:

            lesion_ids.append(row[0])

            image_id = row[1]
            image_path = root + 'images/' + image_id + '.jpg'

            image_paths.append(image_path)
            dxs.append(row[2])


    # open the TFRecords file
    train_writer = tf.python_io.TFRecordWriter(train_filename)
    eval_writer = tf.python_io.TFRecordWriter(eval_filename)


    n_samples = len(image_paths)
    images = []
    labels = []

    all_class_counter = [0]*params["architecture"]["n_output_classes"]


    for i in range(n_samples):

        if not i % 100:
            print('Serializing sample {}/{}'.format(i, n_samples))
            sys.stdout.flush()

        # Load the image
        try:
            pil_img = Image.open(image_paths[i])
        except:
            continue

        # get label of current sample
        label = label_dict[dxs[i]]

        # only allow 100 samples per class
        if all_class_counter[label] > 100:
            continue
        
        all_class_counter[label] += 1

        # image resolution: 600 x 450
        w = 600
        h = 450

        res = params["architecture"]["image_resolution"]

        pil_img = pil_img.crop((75, 0, w-75, h))
        pil_img = pil_img.resize([res, res])

        img = np.asarray(pil_img, dtype=np.uint8)
        img = img.astype(np.float32)
        img = np.multiply(img, 1.0 / 255.0)

        labels.append(label)
        images.append(img)

    combined = list(zip(images, labels))

    # random shuffle images and labels
    shuffle(combined)

    # separate
    images, labels = zip(*combined)

    # split into train, test, and validation set
    training_fraction = 0.8
    training_size = int(round(training_fraction*len(images)))

    # training
    train_images = images[:training_size] 
    train_labels = labels[:training_size] 

    # evaluation
    validation_images = images[training_size:] 
    validation_labels = labels[training_size:] 

    assert(len(validation_images) == len(validation_labels))
    assert(len(train_images) == len(train_labels))

    len_training_examples = len(train_labels)
    len_validation_examples = len(validation_labels)

    print("Training set has {} samples.".format(len_training_examples))
    print("Validation set has {} samples.".format(len_validation_examples))

    # create bins for determining the class distribution in the datasets
    training_class_counter = [0]*params["architecture"]["n_output_classes"]
    val_class_counter = [0]*params["architecture"]["n_output_classes"]

    
    for image, label in zip(validation_images, validation_labels):

        #if label == 1:
        #    continue

        val_class_counter[label] += 1

        # Create a feature
        feature = {'label': _int64_feature(label),
                   'image': _bytes_feature(image.tostring())}

        sample = tf.train.Example(features=tf.train.Features(feature=feature))

        eval_writer.write(sample.SerializeToString())

    for image, label in zip(train_images, train_labels):

        #if label == 1:
        #    continue

        training_class_counter[label] += 1

        # Create a feature
        feature = {'label': _int64_feature(label),
                   'image': _bytes_feature(image.tostring())}

        sample = tf.train.Example(features=tf.train.Features(feature=feature))

        train_writer.write(sample.SerializeToString())


    train_writer.close()
    eval_writer.close()

    print("Distribution of classes in Training set:")
    print(training_class_counter)

    print("Distribution of classes in Evaluation set:")
    print(val_class_counter)


    sys.stdout.flush()


def parse_fn(serialized):

    """Parse TFRecords and perform simple data augmentation."
    """

    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example['image']

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
    print("eval_input_fn: images_batch = ", images_batch)

    images_batch = tf.reshape(images_batch, [1, 16, 16, 3])
    print("eval_input_fn: x = ", images_batch)


    x = {'image': images_batch}
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

    images_batch = tf.reshape(images_batch, [1, 32, 32, 3])
    print("eval_input_fn: x = ", images_batch)

    x = {'image': images_batch}
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

    images_batch = tf.reshape(images_batch, [1, 32, 32, 3])
    print("eval_input_fn: x = ", images_batch)

    x = {'image': images_batch}
    y = labels_batch

    return x, y