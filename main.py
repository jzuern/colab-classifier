from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from data.cifar_dataset import train_input_fn, eval_input_fn, predict_input_fn, maybe_download_and_extract
from model.hyper_parameters import params
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.xception import Xception


tf.logging.set_verbosity(tf.logging.DEBUG)


def resnet_model():

    model = Xception(weights=None,
                      include_top=True,
                      input_shape=(32, 32, 3),
                      classes=10)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=2e-5),
                  metrics=['acc'])

    return model

def main(unused_argv):

    # convert the images to tfrecords
    #maybe_convert_to_tfrecords()

    maybe_download_and_extract()

    load_training_data()
    
    load_validation_data()

    # define a run config
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs=10*60)

    keras_resnet_model = resnet_model()

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

    i = 0
    for result in results:
        i+=1
        if i < 10:
            print('result: {}'.format(result))


if __name__ == "__main__":
    tf.app.run()

