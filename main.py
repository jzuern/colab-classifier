from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from model.resnet_estimator import model_fn
from data.dataset_skin_cancer import train_input_fn, eval_input_fn, maybe_convert_to_tfrecords
import model.hyper_parameters as hyperparams

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    # convert the images to tfrecords
    maybe_convert_to_tfrecords()


    # Create the Estimator
    cancer_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=hyperparams.FLAGS.ckpt_path)

    # define training and evaluation specs
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, 
        max_steps=hyperparams.FLAGS.train_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    # train and evaluate estimator using these specs
    tf.estimator.train_and_evaluate(cancer_classifier, train_spec, eval_spec)




if __name__ == "__main__":
    tf.app.run()

