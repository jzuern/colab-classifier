import tensorflow as tf

from model.hyper_parameters import params
import cv2
import numpy as np
from textwrap import wrap
import re
import itertools
import tfplot
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
import keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3


# from tf.python.keras.layers import Conv2D


def cmatrix_summary(confusion_matrix_tensor_name, labels_names):

    cm = tf.get_default_graph().get_tensor_by_name(confusion_matrix_tensor_name + ':0').eval(session=session).astype(int)

    figure = plot_confusion_matrix(cm, labels_names)
    summary = figure_to_summary(figure)

    return summary


def figure_to_summary(fig, confusion_matrix_tensor_name):
    """
    Converts a matplotlib figure ``fig`` into a TensorFlow Summary object
    that can be directly fed into ``Summary.FileWriter``.
    :param fig: A ``matplotlib.figure.Figure`` object.
    :return: A TensorFlow ``Summary`` protobuf object containing the plot image
             as a image summary.
    """

    # attach a new canvas if not exists
    if fig.canvas is None:
        matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # get PNG data from the figure
    png_buffer = io.BytesIO()
    fig.canvas.print_png(png_buffer)
    png_encoded = png_buffer.getvalue()
    png_buffer.close()

    summary_image = tf.Summary.Image(height=h, width=w, colorspace=4,  # RGB-A
                                  encoded_image_string=png_encoded)

    summary = tf.Summary(value=[tf.Summary.Value(tag=confusion_matrix_tensor_name, image=summary_image)])

    return summary

def plot_confusion_matrix(cm, labels_names):
    '''
    :param cm: A confusion matrix: A square ```numpy array``` of the same size as labels_names
`   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
    '''
    numClasses = len(labels_names)

    fig = matplotlib.figure.Figure(figsize=(numClasses, numClasses), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels_names]
    classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted')
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(numClasses), range(numClasses)):
        ax.text(j, i, int(cm[i, j]) if cm[i, j] != 0 else '.', horizontalalignment="center", verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    return fig


def put_text(imgs, gt_label, pred_label):

    result = np.empty_like(imgs)

    for i in range(imgs.shape[0]):

        text_gt = gt_label[i]
        text_pred = pred_label[i]

        result[i, :, :, :] = cv2.putText(imgs[i, :, :, :], str(text_gt), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 1), 2)
        result[i, :, :, :] = cv2.putText(result[i, :, :, :], str(text_pred), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (1, 0, 0), 2)

    return result


def tf_put_text(imgs, gt_label, pred_label):
    return tf.py_func(put_text, [imgs, gt_label, pred_label], Tout=imgs.dtype)


def vgg_like():

    model = tf.keras.Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(75, 75, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(196, activation='softmax'))


    return model

def mnist():

    l = tf.keras.layers

    input_shape = [75, 75, 1]
    data_format = 'channels_last'

    max_pool = l.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)
    # The model consists of a sequential chain of layers, so tf.keras.Sequential
    # (a subclass of tf.keras.Model) makes for a compact description.
    return tf.keras.Sequential(
        [
            l.Reshape(
                target_shape=input_shape,
                input_shape=(75 * 75,)),
            l.Conv2D(
                32,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Conv2D(
                64,
                5,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.Dropout(0.4),
            l.Dense(196)
        ])


def model_fn(features, labels, mode):

    x = features['image']

    img_res = params["architecture"]["image_resolution"]
    x = tf.reshape(x, [-1, img_res, img_res, 1])
    print("model_fn: x = ", x)


    # model = vgg_like()
    model = mnist()

    logits = model(x)

    predicted_labels = tf.argmax(input=logits, axis=1)

    predictions = {
        "classes": predicted_labels,
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

        optimizer = tf.train.AdamOptimizer(learning_rate=params["training"]["learning_rate"])

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        tf.summary.scalar(name='training/loss', tensor=tf.squeeze(loss))

        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
        
        tf.summary.scalar(name='training/accuracy', tensor=accuracy[1])

        tf.summary.scalar(name='training/gt_labels', tensor=labels[0])

        tf.summary.image(name='training/images', tensor=x, max_outputs=1)

        annotated_images = tf_put_text(x, labels, predicted_labels)
        
        tf.summary.image('training/annotated_images',
                         annotated_images,
                         max_outputs=1)

        ''' confusion matrix summaries '''
        #cm_summary = cmatrix_summary('confusion_matrix', labels_names)

        training_summary_hook = tf.train.SummarySaverHook(
            save_steps=10,
            output_dir=params["paths"]["ckpt_path"],
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks=[training_summary_hook])

    if mode == tf.estimator.ModeKeys.EVAL:

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}


        tf.summary.scalar(name='eval/loss', tensor=tf.squeeze(loss))

        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])


        tf.summary.scalar(name='eval/accuracy', tensor=accuracy[1])

        tf.summary.scalar(name='eval/gt_labels', tensor=labels[0])

        tf.summary.image(name='eval/image', tensor=x, max_outputs=1)

        annotated_images = tf_put_text(x, labels, predicted_labels)
        tf.summary.image('eval/annotated_images',
                         annotated_images,
                         max_outputs=1)

        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=10,
            output_dir=params["paths"]["ckpt_path"],
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=[eval_summary_hook])

    raise ValueError
