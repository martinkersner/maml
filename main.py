from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from data_generator import SinusoidDataGenerator
from trainer import Trainer

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
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot


# def train(
    # model,
    # saver,
    # sess,
    # exp_string,
    # data_generator,
    # resume_itr: int=0,
    # summary_interval: int=100,
    # save_interval: int=1000,
    # print_interval: int=1000,
    # test_print_interval: int=5000,
# ):
    # if FLAGS.log:
        # train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)

    # print('Done initializing, starting training.')
    # prelosses = []
    # postlosses = []
    # multitask_weights = []
    # reg_weights = []

    # num_classes = data_generator.num_classes  # for classification, 1 otherwise

    # for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        # batch_x, batch_y, amp, phase = data_generator.generate()

        # if FLAGS.baseline == 'oracle':
            # batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
            # for i in range(FLAGS.meta_batch_size):
                # batch_x[i, :, 1] = amp[i]
                # batch_x[i, :, 2] = phase[i]

        # feed_dict = {
            # model.input_a: batch_x[:, :num_classes*FLAGS.update_batch_size, :],
            # model.label_a: batch_y[:, :num_classes*FLAGS.update_batch_size, :],

            # # b used for testing
            # model.input_b: batch_x[:, num_classes*FLAGS.update_batch_size:, :],
            # model.label_b: batch_y[:, num_classes*FLAGS.update_batch_size:, :],
        # }

        # if itr < FLAGS.pretrain_iterations:
            # input_tensors = [model.pretrain_op]
        # else:
            # input_tensors = [model.metatrain_op]

        # if (itr % summary_interval == 0 or itr % print_interval == 0):
            # input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            # # if model.classification:
            # #    input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        # result = sess.run(input_tensors, feed_dict)

        # if itr % summary_interval == 0:
            # prelosses.append(result[-2])
            # if FLAGS.log:
                # train_writer.add_summary(result[1], itr)
            # postlosses.append(result[-1])

        # if (itr!=0) and itr % print_interval == 0:
            # if itr < FLAGS.pretrain_iterations:
                # print_str = 'Pretrain Iteration ' + str(itr)
            # else:
                # print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            # print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            # print(print_str)
            # prelosses, postlosses = [], []

        # if (itr!=0) and itr % save_interval == 0:
            # saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # # sinusoid is infinite data, so no need to test on meta-validation set.
        # if (itr!=0) and itr % test_print_interval == 0 and FLAGS.datasource !='sinusoid':
            # if 'generate' not in dir(data_generator):
                # feed_dict = {}
                # if model.classification:
                    # input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                # else:
                    # input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            # else:
                # batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                # input_a = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                # input_b = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                # label_a = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                # label_b = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                # feed_dict = {
                    # model.input_a: input_a,
                    # model.input_b: input_b,
                    # model.label_a: label_a,
                    # model.label_b: label_b,
                    # model.meta_lr: 0.0
                # }
                # if model.classification:
                    # input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                # else:
                    # input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

            # result = sess.run(input_tensors, feed_dict)
            # print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    # saver.save(sess, Path(FLAGS.logdir) / exp_string / f"model{str(itr)}")


def main():
    # if FLAGS.train:
        # test_num_updates = 5
    # else:
        # test_num_updates = 10
        # orig_meta_batch_size = FLAGS.meta_batch_size
        # # always use meta batch size of 1 when testing.
        # FLAGS.meta_batch_size = 1

    data_generator = SinusoidDataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    # dim_output = data_generator.dim_output

    # if FLAGS.baseline == 'oracle':
        # assert FLAGS.datasource == 'sinusoid'
        # dim_input = 3
        # FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        # FLAGS.metatrain_iterations = 0
    # else:
        # dim_input = data_generator.dim_input

    # model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)

    # if FLAGS.train:
    # model.construct_model(input_tensors=None, prefix='metatrain_')

    # model.summ_op = tf.summary.merge_all()

    # saver = tf.train.Saver(
        # tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    # sess = tf.InteractiveSession()

    # if not FLAGS.train:
        # # change to original meta batch size when loading model.
        # FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = build_exp_string()

    # resume_itr = 0
    # model_file = None

    # tf.global_variables_initializer().run()
    # tf.train.start_queue_runners()

    # if FLAGS.resume or not FLAGS.train:
        # model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        # if FLAGS.test_iter > 0:
            # model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        # if model_file:
            # ind1 = model_file.index('model')
            # resume_itr = int(model_file[ind1+5:])
            # print("Restoring model weights from " + model_file)
            # saver.restore(sess, model_file)

    trainer = Trainer(
        data_generator,
        exp_string,
        Path(FLAGS.logdir),
        FLAGS.pretrain_iterations,
        FLAGS.metatrain_iterations,
        FLAGS.meta_batch_size,
        FLAGS.update_batch_size,
        FLAGS.num_updates,
        baseline=FLAGS.baseline,
        is_training=FLAGS.train,
    )

    trainer.train()

    # if FLAGS.train:
        # train(model, saver, sess, exp_string, data_generator, resume_itr)
    # else:
        # test(model, saver, sess, exp_string, data_generator, test_num_updates)

def build_exp_string():
    exp_string = "".join(
        [
        f"cls_{str(FLAGS.num_classes)}",
        f".mbs_{str(FLAGS.meta_batch_size)}",
        f".ubs_{str(FLAGS.train_update_batch_size)}",
        f".numstep_{str(FLAGS.num_updates)}",
        f".updatelr_{str(FLAGS.train_update_lr)}"
        ]
    )

    if FLAGS.num_filters != 64:
        exp_string += f"hidden_{str(FLAGS.num_filters)}"
    if FLAGS.max_pool:
        exp_string += "maxpool"
    if FLAGS.stop_grad:
        exp_string += "stopgrad"
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == "batch_norm":
        exp_string += "batchnorm"
    elif FLAGS.norm == "layer_norm":
        exp_string += "layernorm"
    elif FLAGS.norm == "None":
        exp_string += "nonorm"
    else:
        print("Norm setting not recognized.")

    return exp_string


if __name__ == "__main__":
    main()
