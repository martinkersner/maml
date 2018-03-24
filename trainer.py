from pathlib import Path
import logging

import numpy as np
import tensorflow as tf

from maml import MAML
from utils import SummaryFileWriter, TrainSaver


class Trainer():
    def __init__(
        self,
        data_generator,
        exp_string,  # move out
        logdir: Path,
        pretrain_iterations,
        metatrain_iterations,
        meta_batch_size,
        update_batch_size,
        num_updates,
        baseline: str=None,
        is_training: bool=True,
    ):
        self.data_generator = data_generator
        self.exp_string = exp_string
        self.logdir = logdir
        self.pretrain_iterations = pretrain_iterations
        self.metatrain_iterations = metatrain_iterations
        self.meta_batch_size = meta_batch_size
        self.update_batch_size = update_batch_size
        self.num_updates = num_updates
        self.baseline = baseline

        self.summary_interval = 100
        self.save_interval = 100
        self.print_interval = 100
        self.test_print_interval = 5000

        self.session = tf.InteractiveSession()
        self.train_writer = SummaryFileWriter(self.logdir / self.exp_string, self.session.graph)

        self.num_classes = self.data_generator.num_classes  # for classification, 1 otherwise

        self.build_model(is_training)

    def build_model(self, is_training: bool):
        if is_training:
            test_num_updates = 5
        else:
            test_num_updates = 10
            orig_meta_batch_size = self.meta_batch_size

            self.meta_batch_size = 1  # always use meta batch size of 1 when testing.

        if self.baseline == 'oracle':
            assert FLAGS.datasource == 'sinusoid'
            dim_input = 3
            self.pretrain_iterations += self.metatrain_iterations
            self.metatrain_iterations = 0
        else:
            dim_input = self.data_generator.dim_input

        dim_output = 1

        self.model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
        self.model.construct_model(input_tensors=None, prefix='metatrain_')
        self.model.summ_op = tf.summary.merge_all()

        tf.global_variables_initializer().run()
        self.saver = TrainSaver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def train(self):
        logging.info("Initialization")
        tf.train.start_queue_runners()

        logging.info("Training started")
        prelosses = []
        postlosses = []
        multitask_weights = []
        reg_weights = []

        for itr in range(self.pretrain_iterations + self.metatrain_iterations):
            batch_x, batch_y, amp, phase = self.data_generator.generate()

            # TODO test
            if self.baseline == 'oracle':
                batch_x = np.concatenate(
                    [
                        batch_x,
                        np.zeros([batch_x.shape[0], batch_x.shape[1], 2])
                    ],
                    2
                )

                for i in range(self.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            feed_dict = {
                self.model.input_a: batch_x[:, :self.num_classes*self.update_batch_size, :],
                self.model.label_a: batch_y[:, :self.num_classes*self.update_batch_size, :],

                # b used for testing
                self.model.input_b: batch_x[:, self.num_classes*self.update_batch_size:, :],
                self.model.label_b: batch_y[:, self.num_classes*self.update_batch_size:, :],
            }

            if itr < self.pretrain_iterations:
                input_tensors = [self.model.pretrain_op]
            else:
                input_tensors = [self.model.metatrain_op]

            if (itr % self.summary_interval == 0 or itr % self.print_interval == 0):
                input_tensors.extend(
                    [
                        self.model.summ_op,
                        self.model.total_loss1,
                        self.model.total_losses2[self.num_updates-1]
                    ])
                # if model.classification:
                #    input_tensors.extend([model.total_accuracy1, model.total_accuracies2[self.num_updates-1]])

            result = self.session.run(input_tensors, feed_dict)

            if itr % self.summary_interval == 0:
                prelosses.append(result[-2])
                self.train_writer.add_summary(result[1], itr)
                postlosses.append(result[-1])

            if itr != 0 and itr % self.print_interval == 0:
                if itr < self.pretrain_iterations:
                    print_str = 'Pretrain Iteration ' + str(itr)
                else:
                    print_str = 'Iteration ' + str(itr - self.pretrain_iterations)
                print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
                print(print_str)
                prelosses, postlosses = [], []

            if itr != 0 and itr % self.save_interval == 0:
                self.saver.save(self.session, self.logdir / self.exp_string / f"model{str(itr)}")

            # sinusoid is infinite data, so no need to test on meta-validation set.
            # if itr != 0 and itr % test_print_interval == 0 and FLAGS.datasource !='sinusoid':
            #     if 'generate' not in dir(data_generator):
            #         feed_dict = {}
            #         if model.classification:
            #             input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[self.num_updates-1], model.summ_op]
            #         else:
            #             input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[self.num_updates-1], model.summ_op]
            #     else:
            #         batch_x, batch_y, amp, phase = data_generator.generate(train=False)
            #         input_a = batch_x[:, :num_classes*self.update_batch_size, :]
            #         input_b = batch_x[:, num_classes*self.update_batch_size:, :]
            #         label_a = batch_y[:, :num_classes*self.update_batch_size, :]
            #         label_b = batch_y[:, num_classes*self.update_batch_size:, :]
            #         feed_dict = {
            #             model.input_a: input_a,
            #             model.input_b: input_b,
            #             model.label_a: label_a,
            #             model.label_b: label_b,
            #             model.meta_lr: 0.0
            #         }
            #         if model.classification:
            #             input_tensors = [model.total_accuracy1, model.total_accuracies2[self.num_updates-1]]
            #         else:
            #             input_tensors = [model.total_loss1, model.total_losses2[self.num_updates-1]]

            #     result = sess.run(input_tensors, feed_dict)
            #     logging.info(f"Validation results: {str(result[0])}, {str(result[1]}")

        self.saver.save(self.session, self.logdir / self.exp_string / f"model{str(itr)}")
