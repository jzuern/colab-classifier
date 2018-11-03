

import numpy as np
import pickle
import os
import urllib
from model.hyper_parameters import params
########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = params["paths"]["data_dir"]

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.





def download(base_url, filename, download_dir):
    """
    Download the given file if it does not already exist in the download_dir.
    :param base_url: The internet URL without the filename.
    :param filename: The filename that will be added to the base_url.
    :param download_dir: Local directory for storing the file.
    :return: Nothing.
    """

    # Path for local file.
    save_path = os.path.join(download_dir, filename)

    # Check if the file already exists, otherwise we need to download it now.
    if not os.path.exists(save_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        print("Downloading", filename, "...")

        # Download the file from the internet.
        url = base_url + filename
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=save_path)

        print(" Done!")


def maybe_download_and_extract(url=data_url, download_dir=data_path):
    """
    Download and extract the data if it doesn't already exist.
    Assumes the url is a tar-ball file.
    :param url:
        Internet URL for the tar-file to download.
        Example: "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    :param download_dir:
        Directory where the downloaded file is saved.
        Example: "data/CIFAR-10/"
    :return:
        Nothing.
    """

    # Filename for saving the file downloaded from the internet.
    # Use the filename from the URL and add it to the download_dir.
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    # Check if the file already exists.
    # If it exists then we assume it has also been extracted,
    # otherwise we need to download and extract it now.
    if not os.path.exists(file_path):
        # Check if the download directory exists, otherwise create it.
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download the file from the internet.
        file_path, _ = urllib.request.urlretrieve(url=url,
                                                  filename=file_path)

        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            # Unpack the zip-file.
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            # Unpack the tar-ball.
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded and unpacked.")


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parse_fn(serialized):

    """Parse TFRecords and perform simple data augmentation."
    """

    features = \
        {
            str(params["architecture"]["image_input_name"]): tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    image_raw = parsed_example[str(params["architecture"]["image_input_name"])]

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.float32)
    # print("parse_fn: image = ", image)

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

    x = {str(params["architecture"]["image_input_name"]): images_batch}
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

    dataset = dataset.shuffle(buffer_size=buffer_size)


    # If testing then don't shuffle the data.
    # Only go through the data once.
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
    res = params["architecture"]["image_resolution"]

    images_batch = tf.reshape(images_batch, [-1, res, res, 3])

    x = {str(params["architecture"]["image_input_name"]): images_batch}
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

    images_batch = tf.reshape(images_batch, [-1, res, res, 3])

    x = {str(params["architecture"]["image_input_name"]): images_batch}
    y = labels_batch

    return x, y





def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.

    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, "cifar-10-batches-py/", filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.

    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])




    # Convert the images.
    images = _convert_images(raw_images)


    for image, cl in zip(images, cls):

        # Create a feature
        feature = {'label': _int64_feature(label),
               str(params["architecture"]["image_input_name"]): _bytes_feature(img.tostring())}

        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(sample.SerializeToString())




    return images, cls


def load_class_names():
    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.

    The data-set is split into 5 data-files which are merged here.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    # write to tfrecords:
    # open the TFRecords file
    root = params["paths"]["data_dir"]

    train_writer = tf.python_io.TFRecordWriter(root + 'train.tfrecords')

    for i in range(_num_images_train):

        label = cls[i]
        img = images[i, :]

        # Create a feature
        feature = {'label': _int64_feature(label),
                   str(params["architecture"]["image_input_name"]): _bytes_feature(img.tostring())}

        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(sample.SerializeToString())


    # return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.

    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch")

    # write to tfrecords:
    # open the TFRecords file
    root = params["paths"]["data_dir"]

    train_writer = tf.python_io.TFRecordWriter(root + 'eval.tfrecords')

    for i in range(_num_images_train):


        # Create a feature
        feature = {'label': _int64_feature(label),
                   str(params["architecture"]["image_input_name"]): _bytes_feature(img.tostring())}

        sample = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(sample.SerializeToString())



    #return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

########################################################################
