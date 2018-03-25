import tensorflow as tf

from utils import mse, normalize


class MAML(object):
    def __init__(
        self,
        dim_input: int=1,
        dim_output: int=1,
        num_updates: int=1,
        update_lr: float=1e-3,
        meta_lr: float=1e-3,
        test_num_updates: int=5,
        stop_grad: bool=False,
        meta_batch_size: int=25,
        metatrain_iterations: int=15000,
        norm: str="batch_norm",
    ):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_updates = num_updates
        self.update_lr = update_lr
        self.meta_lr = tf.placeholder_with_default(meta_lr, ())
        self.test_num_updates = test_num_updates
        self.stop_grad = stop_grad
        self.meta_batch_size = meta_batch_size
        self.metatrain_iterations = metatrain_iterations
        self.norm = norm

        self.dim_hidden = [40, 40]
        self.dims = [self.dim_input, 40, 40, self.dim_output]
        self.loss_func = mse
        self.forward = self.forward_fc
        self.initialize_weights = self.initialize_fc_weights

    def build(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient
        # b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)  # FIXME rename
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        # else:
            # self.inputa = input_tensors['inputa']
            # self.inputb = input_tensors['inputb']
            # self.labela = input_tensors['labela']
            # self.labelb = input_tensors['labelb']

        # with tf.variable_scope('model', reuse=None) as training_scope:
        with tf.variable_scope('model', reuse=None):
            # if 'weights' in dir(self):  # TODO correct?
                # training_scope.reuse_variables()
                # weights = self.weights
            # else:
                # Define the weights
            self.weights = weights = self.initialize_weights()

            self.num_updates_tmp = max(self.test_num_updates, self.num_updates)
            task_metalearn = self.metalearn_wrapper()

            if self.norm is not "None":  # FIXME
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*self.num_updates_tmp, tf.float32, [tf.float32]*self.num_updates_tmp]

            outputs_a, outputs_b, losses_a, losses_b = tf.map_fn(
                task_metalearn,
                elems=(self.inputa, self.inputb, self.labela, self.labelb),
                dtype=out_dtype,
                parallel_iterations=self.meta_batch_size
            )

        if "train" in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(losses_a) / tf.to_float(self.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(losses_b[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates_tmp)]
            # after the map_fn
            # self.outputs_a, self.outputs_b = outputs_a, outputs_b
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr, name="Adam/pretrain").minimize(total_loss1)

            if self.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr, name="Adam/metatrain")
                self.gvs = optimizer.compute_gradients(self.total_losses2[-1])
                self.metatrain_op = optimizer.apply_gradients(self.gvs)
        else:  # metaval_
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(losses_a) / tf.to_float(self.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(losses_b[j]) / tf.to_float(self.meta_batch_size) for j in range(self.num_updates_tmp)]

        tf.summary.scalar(f"{prefix}/pre_update_loss", total_loss1)

        for j in range(self.num_updates_tmp):
            tf.summary.scalar(f"{prefix}/post_update_loss/step_{j+1}", total_losses2[j])

        self.summary_op = tf.summary.merge_all()

    def metalearn_wrapper(self):
        """ Perform gradient descent for one task in the meta-batch. """
        def fun(inp, reuse: bool=True):
            with tf.variable_scope("model", reuse=True):
                inputa, inputb, labela, labelb = inp

                task_outputs_b = []
                task_losses_b = []

                fast_weights = self.weights

                for i in range(self.num_updates_tmp):
                    if i == 0:
                        reuse_tmp = reuse
                    else:
                        reuse_tmp = True

                    output_a = self.forward(inputa, fast_weights, reuse=reuse_tmp)
                    loss_a = self.loss_func(output_a, labela)

                    if i == 0:
                        task_output_a = output_a
                        task_loss_a = loss_a

                    grads = tf.gradients(loss_a, list(fast_weights.values()))
                    if self.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))

                    output_b = self.forward(inputb, fast_weights, reuse=True)
                    loss_b = self.loss_func(output_b, labelb)
                    task_outputs_b.append(output_b)
                    task_losses_b.append(loss_b)

                return [task_output_a, task_outputs_b, task_loss_a, task_losses_b]

        return fun

    def initialize_fc_weights(self):
        weights = {}

        for idx in range(len(self.dims)-1):
            weights[f"w{idx}"] = tf.Variable(tf.truncated_normal([self.dims[idx], self.dims[idx+1]], stddev=0.01))
            weights[f"b{idx}"] = tf.Variable(tf.zeros([self.dims[idx+1]]))

        return weights

    def forward_fc(self, hidden, weights, reuse: bool=False):
        num_op = len(self.dims) - 1
        last_op = num_op -1

        for idx in range(num_op):
            hidden = tf.matmul(hidden, weights[f"w{idx}"]) + weights[f"b{idx}"]
            if idx != last_op:
                hidden = normalize(hidden, activation=tf.nn.relu, reuse=reuse, scope=f"norm{idx}")

        return hidden
