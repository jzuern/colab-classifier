
from tensorflow.contrib.training import HParams

hparams = HParams(
    n_classes=10,
    learning_rate=5e-4,
    image_resolution=32,
    train_batch_size=1,
    val_batch_size=1,
    test_batch_size=1,
    n_train_steps=10000,
    input_name='input_1',
    data_dir='/tmp/cifar-data/',
    checkpoint_dir='/tmp/checkpoints/'
)

