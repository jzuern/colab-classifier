from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from data.car_dataset import train_input_fn, eval_input_fn, predict_input_fn, maybe_convert_to_tfrecords
from model.hyper_parameters import params
from tensorflow.python.keras.applications.resnet50 import ResNet50


tf.logging.set_verbosity(tf.logging.DEBUG)


def resnet_model():

    model = ResNet50(weights=None,
                      include_top=True,
                      input_shape=(75, 75, 3),
                      classes=196)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

    return model

def main(unused_argv):

    # convert the images to tfrecords
    maybe_convert_to_tfrecords()

    # define a run config
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs=2*60)

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
        throttle_secs=60)

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

