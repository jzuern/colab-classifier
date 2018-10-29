# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
import tensorflow as tf
import json

with open('/content/drive/classifier/params.json') as f:
    params = json.load(f)


# FLAGS = tf.app.flags.FLAGS


## The following flags define hyper-parameters regards training

# tf.app.flags.DEFINE_float('learning_rate', 0.0001, '''Learning rate of optimizer''')
# tf.app.flags.DEFINE_integer('train_steps', 10000, '''Total steps that you want to train''')
# tf.app.flags.DEFINE_integer('train_batch_size', 32, '''Train batch size''')
# tf.app.flags.DEFINE_integer('validation_batch_size', 32, '''Validation batch size''')
# tf.app.flags.DEFINE_integer('test_batch_size', 32, '''Test batch size''')

# ## The following flags define hyper-parameters modifying the training network

# tf.app.flags.DEFINE_integer('num_residual_blocks', 6, '''How many residual blocks do you want''')
# tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')

# ## If you want to load a checkpoint and continue training

# tf.app.flags.DEFINE_string('data_dir', '/content/dataset_skin/', '''Checkpoint directory to restore''')
# tf.app.flags.DEFINE_string('ckpt_path', '/content/drive/classifier/checkpoints', '''Checkpoint directory to restore''')
# tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continuetraining''')

# tf.app.flags.DEFINE_integer('image_resolution', 64, '''Resolution of images in x and y dimensions''')
# tf.app.flags.DEFINE_string('activation_function', 'relu', '''Resolution of images in x and y dimensions''')
