import csv
import os
import tensorflow as tf
import sys
from PIL import Image
import numpy as np
import model.hyper_parameters as hyperparams


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def maybe_convert_to_tfrecords():


    root = hyperparams.FLAGS.data_dir

    # Test if tfrecords files already exist
    train_filename = root + 'train.tfrecords'
    eval_filename = root + 'eval.tfrecords'

    if os.path.isfile(train_filename) and os.path.isfile(eval_filename):
        print("Training and Evaluation tfrecord files already exist!")
        return

    # if tfrecords files don't exist:

    label_dict = {"bkl": 0,
                  "nv": 1,
                  "mel": 2,
                  "bcc": 3,
                  "akiec": 4,
                  "vasc": 5,
                  "df": 6}

    csv_file = root + 'HAM10000_metadata.csv'

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

    max_samples = len(image_paths)

    for i in range(max_samples):

        if not i % 100:
            print('Serializing sample {}/{}'.format(i, max_samples))
            sys.stdout.flush()

        # Load the image
        try:
            pil_img = Image.open(image_paths[i])
        except:
            continue


        # image resolution: 600 x 450
        w = 600
        h = 450

        pil_img = pil_img.crop((75, 0, w-75, h))
        pil_img = pil_img.resize([hyperparams.FLAGS.image_resolution, hyperparams.FLAGS.image_resolution])

        img = np.asarray(pil_img, dtype=np.uint8)
        img = img.astype(np.float32)
        img = np.multiply(img, 1.0 / 255.0)


        label = label_dict[dxs[i]]

        # Create a feature
        feature = {'label': _int64_feature(label),
                   'image': _bytes_feature(img.tostring())}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        if i % 5 == 0:
            eval_writer.write(example.SerializeToString())
        else:
            train_writer.write(example.SerializeToString())

    train_writer.close()
    eval_writer.close()

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

    image = tf.reshape(image, [hyperparams.FLAGS.image_resolution, hyperparams.FLAGS.image_resolution, 3])

    # Get the label associated with the image.
    label = parsed_example['label']

    # The image and label are now correct TensorFlow types.

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly flip the image horizontally and vertically.
    #image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_flip_up_down(image)

    return image, label


def train_input_fn(batch_size=hyperparams.FLAGS.train_batch_size, buffer_size=100000):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    tfrecords_filename = hyperparams.FLAGS.data_dir + 'train.tfrecords'

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
    x = {'image': images_batch}
    y = labels_batch

    return x, y


def eval_input_fn(batch_size=hyperparams.FLAGS.validation_batch_size, buffer_size=100000):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    tfrecords_filename = hyperparams.FLAGS.data_dir + 'eval.tfrecords'

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_filename)

    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.

    dataset = dataset.map(parse_fn)

    # If testing then don't shuffle the data.
    # Only go through the data once.
    num_repeat = None

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Maximum number of elements that will be buffered
    # prefetch(n) (where n is the number of elements / batches consumed by a training step)

    prefetch_buffer_size = 100

    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    x = {'image': images_batch}
    y = labels_batch

    return x, y


if __name__ == '__main__':

    convert_to_tfrecords()
