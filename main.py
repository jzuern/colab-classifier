from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from data.cifar_dataset import train_input_fn, eval_input_fn, predict_input_fn, maybe_download_and_extract, load_training_data, load_validation_data
from model.hyper_parameters import params
from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import Sequential
import argparse

tf.logging.set_verbosity(tf.logging.DEBUG)


def cnn_model():

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  metrics=['acc'])

    return model

def main(unused_argv):

    # convert the images to tfrecords
    maybe_download_and_extract()

    load_training_data()
    
    load_validation_data()

    # define a run config
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs=10*60)

    keras_resnet_model = cnn_model()

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=keras_resnet_model,
        model_dir=params["paths"]["ckpt_path"],
        config=run_config)

    # define training and evaluation specs
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, 
        max_steps=params["training"]["train_steps"])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        throttle_secs=10*60)

    # train and evaluate estimator using these specs
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # predict class based on input
    results = estimator.predict(input_fn=predict_input_fn)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #
    # parser.add_argument('checkpoint_dir', default="/tmp/checkpoints",
    #                     help='Directory of the checkpoint')
    # parser.add_argument('data_dir', default="/tmp/cifar-10-data",
    #                     help='Directory to save the CIFAR-10 data')

    args = parser.parse_args()

    tf.app.run()

