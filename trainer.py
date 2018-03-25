from pathlib import Path
import logging

import numpy as np
import tensorflow as tf

from maml import MAML
from utils import SummaryFileWriter, TrainSaver

logging.basicConfig(level=logging.INFO)


class Trainer(object):
    def __init__(
        self,
        model,
        data_generator,
        logdir: Path,
        pretrain_iterations,
        metatrain_iterations,
        meta_batch_size,
        update_batch_size,
        num_updates,
        baseline: str=None,
        stop_grad: bool=False,
        is_training: bool=True,
    ):
        self.model = model
        self.data_generator = data_generator
        self.logdir = logdir
        self.pretrain_iterations = pretrain_iterations
        self.metatrain_iterations = metatrain_iterations
        self.meta_batch_size = meta_batch_size
        self.update_batch_size = update_batch_size
        self.num_updates = num_updates
        self.baseline = baseline
        self.stop_grad = stop_grad
        self.is_training = is_training

        # for classification, 1 otherwise
        self.num_classes = self.data_generator.num_classes

        self.setup()

    def setup(self):
        self.log_interval = 100
        self.save_interval = 1000
        self.exp_string = self._build_exp_string()

        self.session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        self.saver = TrainSaver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        self.summary_writer = SummaryFileWriter(self.logdir / self.exp_string,
                                                self.session.graph)

    def train(self):
        logging.info("Training started")
        self.prelosses = []
        self.postlosses = []

        for step in range(self.pretrain_iterations + self.metatrain_iterations):
            batch_x, batch_y, amp, phase = self.data_generator.generate()  # FIXME amp, phase

            # b used for testing
            feed_dict = {
                self.model.inputa: batch_x[:, :self.num_classes*self.update_batch_size, :],
                self.model.inputb: batch_x[:, self.num_classes*self.update_batch_size:, :],
                self.model.labela: batch_y[:, :self.num_classes*self.update_batch_size, :],
                self.model.labelb: batch_y[:, self.num_classes*self.update_batch_size:, :],
            }

            if step < self.pretrain_iterations:
                train_op = self.model.pretrain_op
            else:
                train_op = self.model.metatrain_op

            input_tensors = [
                train_op,
                self.model.summ_op,
                self.model.total_loss1,
                self.model.total_losses2[self.num_updates-1]
            ]

            _, summary, preloss, postloss = self.session.run(input_tensors, feed_dict)
            self.prelosses.append(preloss)
            self.postlosses.append(postloss)

            if step != 0:
                if step % self.log_interval == 0:
                    self.summary_writer.add_summary(summary, global_step=step)
                    self._log_training_info(step)
                    self.prelosses = []
                    self.postlosses = []

                if step % self.save_interval == 0:
                    self.saver.save(self.session, self.logdir / self.exp_string / f"model{step}")

        self.saver.save(self.session, self.logdir / self.exp_string / f"model{step}")

    def _log_training_info(self, step):
        if step < self.pretrain_iterations:
            log_str = f"Pretrain step {step}"
        else:
            log_str = f"Step {step - self.pretrain_iterations}"

        log_str += f": {np.mean(self.prelosses)}, {np.mean(self.postlosses)}"
        logging.info(log_str)

    def _build_exp_string(self):
        exp_string = "".join([
            f"cls_{str(self.num_classes)}",
            f".mbs_{str(self.meta_batch_size)}",
            f".numstep_{str(self.num_updates)}"])

        # if FLAGS.num_filters != 64:
            # exp_string += f"hidden_{str(FLAGS.num_filters)}"
        # if FLAGS.max_pool:
            # exp_string += "maxpool"
        if self.stop_grad:
            exp_string += "stopgrad"
        if self.baseline:
            exp_string += FLAGS.baseline

        # if FLAGS.norm == "batch_norm":
            # exp_string += "batchnorm"
        # elif FLAGS.norm == "layer_norm":
            # exp_string += "layernorm"
        # elif FLAGS.norm == "None":
            # exp_string += "nonorm"
        # else:
            # raise ValueError("Norm setting not recognized.")

        return exp_string
