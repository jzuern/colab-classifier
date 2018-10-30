from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from model.resnet_estimator import model_fn
from data.dataset_skin_cancer import train_input_fn, eval_input_fn, predict_input_fn, maybe_convert_to_tfrecords
from model.hyper_parameters import params

tf.logging.set_verbosity(tf.logging.DEBUG)


def main(unused_argv):

    # convert the images to tfrecords
    maybe_convert_to_tfrecords()

    # define a run config
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs=120)

    # Create the Estimator
    cancer_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params["paths"]["ckpt_path"],
        config=run_config)

    # define training and evaluation specs
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, 
        max_steps=params["training"]["train_steps"])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        throttle_secs=100)

    # train and evaluate estimator using these specs
    tf.estimator.train_and_evaluate(cancer_classifier, train_spec, eval_spec)

    # predict class based on input
    results = cancer_classifier.predict(input_fn=predict_input_fn)

    i = 0
    for result in results:
        i+=1
        if i < 10:
            print('result: {}'.format(result))


if __name__ == "__main__":
    tf.app.run()

