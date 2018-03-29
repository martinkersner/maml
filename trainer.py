from pathlib import Path
import logging
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pickle
import csv

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
        update_lr,
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
        self.update_lr = update_lr
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
            feed_dict = self.get_feed_dict(is_training=True)
            # batch_x, batch_y, amp, phase = self.data_generator.generate()  # FIXME amp, phase

            # b used for testing
            # feed_dict = {
                # self.model.input_a: batch_x[:, :self.num_classes*self.update_batch_size, :],
                # self.model.input_b: batch_x[:, self.num_classes*self.update_batch_size:, :],
                # self.model.label_a: batch_y[:, :self.num_classes*self.update_batch_size, :],
                # self.model.label_b: batch_y[:, self.num_classes*self.update_batch_size:, :],
            # }

            if step < self.pretrain_iterations:
                train_op = self.model.pretrain_op
            else:
                train_op = self.model.metatrain_op

            input_tensors = [
                train_op,
                self.model.summary_op,
                self.model.total_loss1,
                # self.model.total_losses2[self.num_updates-1]
                self.model.total_losses2[-1]
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
        logging.info("Training finished")

    def test(
        self,
        num_test_points: int=10  # TODO FIXME
    ):
        logging.info("Testing started")

        # TODO move to  generator
        np.random.seed(1)
        random.seed(1)

        metaval_accuracies = []

        for _ in tqdm(range(num_test_points)):
            # if 'generate' not in dir(data_generator):
                # feed_dict = {}
                # feed_dict = {model.meta_lr : 0.0}
            # else:
            feed_dict = self.get_feed_dict(is_training=False)

            # if model.classification:
                # result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
            # else:  # this is for sinusoid

            # loss_1, loss_2_1, loss_2_2, loss_2_3, loss_2_4,, loss_2_5
            result = self.session.run([self.model.total_loss1] + self.model.total_losses2, feed_dict)
            metaval_accuracies.append(result)  # FIXME not accuracies

        from IPython import embed; embed()  # XXX DEBUG

        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96*stds/np.sqrt(num_test_points)

        logging.info('Mean validation accuracy/loss, stddev, and confidence intervals')
        logging.info(means)
        logging.info(stds)
        logging.info(ci95)

        tmp_name = self.logdir / self.exp_string / f"test_ubs{self.update_batch_size}_stepsize{self.update_lr}"
        out_filename = tmp_name.with_suffix(".csv")
        out_pkl = tmp_name.with_suffix(".pkl")
        from IPython import embed; embed()  # XXX DEBUG

        with open(out_pkl, "wb") as f:
            pickle.dump({"mses": metaval_accuracies}, f)
        with open(out_filename, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["update"+str(i) for i in range(len(means))])
            writer.writerow(means)
            writer.writerow(stds)
            writer.writerow(ci95)

    def get_feed_dict(self, is_training: bool=False):
        batch_x, batch_y, amp, phase = self.data_generator.generate(is_training=is_training)

        feed_dict = {}

        if not is_training and self.baseline == "oracle":
            batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
            batch_x[0, :, 1] = amp[0]
            batch_x[0, :, 2] = phase[0]
            feed_dict[self.model_meta_lr] = 0.0

        feed_dict[self.model.input_a] = batch_x[:, :self.num_classes*self.update_batch_size, :]
        feed_dict[self.model.input_b] = batch_x[:, self.num_classes*self.update_batch_size:, :]
        feed_dict[self.model.label_a] = batch_y[:, :self.num_classes*self.update_batch_size, :]
        feed_dict[self.model.label_b] = batch_y[:, self.num_classes*self.update_batch_size:, :]

        return feed_dict

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
