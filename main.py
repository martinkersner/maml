from pathlib import Path
import numpy as np
import tensorflow as tf

from data_generator import SinusoidDataGenerator
from maml import MAML
from trainer import Trainer
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot


def main():
    if FLAGS.train:
        test_num_updates = 5
    else:
        test_num_updates = 10

        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    data_generator = SinusoidDataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)

    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input

    input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    model.summ_op = tf.summary.merge_all()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    trainer = Trainer(
        model,
        data_generator,
        Path(FLAGS.logdir),
        FLAGS.pretrain_iterations,
        FLAGS.metatrain_iterations,
        FLAGS.meta_batch_size,
        FLAGS.update_batch_size,
        FLAGS.num_updates,
        stop_grad=FLAGS.stop_grad,
        baseline=FLAGS.baseline,
        is_training=True
    )

    trainer.train()


if __name__ == "__main__":
    main()
