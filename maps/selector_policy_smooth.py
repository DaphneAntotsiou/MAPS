import tensorflow as tf
import numpy as np
import maps.tf_util as U


class SelectorPolicy(object):
    def __init__(self, name, module_num, task_space, ob_space, hidden_size, loss=True,
                 task_num=None, c_sharing=1, c_sparsity=1, c_exploration=1, c_smooth=1):
        self.task_shape = task_space.shape
        self.state_shape = ob_space.shape
        self.hidden_size = hidden_size
        self.module_num = module_num
        self.task_num = task_num
        # state tensor will be shared with modules, hence outside the scope
        self.ob_ph = U.get_placeholder(dtype=tf.float32, shape=[None] + list(self.state_shape), name="ob")

        with tf.variable_scope(name, reuse=False):
            self.scope = tf.get_variable_scope().name
            self._init(c_sharing, c_sparsity, c_exploration, c_smooth, loss=loss)

    def _init(self, c_sharing=1, c_sparsity=1, c_exploration=1, c_smooth=1, loss=True):
        self.task_ph = tf.placeholder(tf.float32, [None] + list(self.task_shape), name="task_ph")   # needs to be float to match obs
        _input = tf.concat([self.ob_ph, self.task_ph], axis=1)  # concatenate the two input -> form a transition

        self.orig_logits = self.logits = self._build_graph(_input, reuse=False)

        self.losses = []
        if loss and self.task_num:
            L_sparsity = tf.reduce_mean(tf.reduce_mean(-tf.pow(self.logits+1e-8, 1/tf.log(np.float32(self.module_num))) * tf.log(self.logits+1e-8), axis=1))  # entropy Sx^(1/logM)logx/M
            L_sharing = tf.reduce_mean(self._sharing_loss())
            L_exploration = self._exploration_loss(self.logits)
            self.prev_ob_ph = U.get_placeholder(dtype=tf.float32, shape=[None] + list(self.state_shape), name="prev_ob")
            L_smooth = self._smooth_loss()
            self.total_loss = (c_sparsity * L_sparsity + c_sharing * L_sharing + c_exploration * L_exploration +
                               c_smooth * L_smooth)
            self.losses = [L_sparsity, L_sharing, L_exploration, L_smooth]
            self.loss_name = ["sparsity_loss", "sharing_loss", "exploration_loss", "smoothing_loss"]

        self._output = U.function([self.ob_ph, self.task_ph], [self.logits])

    def output(self, state, task):
        if not isinstance(task, np.ndarray):
            task = np.array(task)
        output = self._output(state[None], task[None])
        return np.squeeze(output[0])

    @staticmethod
    def _exploration_loss(logits):
        # sum batch and maximise coefficient of variation
        m, var = tf.nn.moments(tf.reduce_sum(logits, axis=0), axes=[0])
        loss = tf.div(var, tf.square(m))
        return loss

    def _sharing_loss(self):
        # L sharing needs multiple outputs for all task inputs
        task_logits = []    # logits with all possible task inputs
        for i in range(self.task_num):
            if self.task_shape[0] == 1:
                fixed_task = tf.fill(tf.shape(self.task_ph), np.float32(i))
            else:
                fixed_task = tf.constant([1 if i == el else 0 for el in range(self.task_num)],
                                         dtype=tf.float32)
                fixed_task = tf.broadcast_to(fixed_task, shape=tf.shape(self.task_ph))

            _input = tf.concat([self.ob_ph, fixed_task], axis=1)  # concatenate the two input -> form a transition
            task_logits.append(self._build_graph(_input, reuse=True))

        assert task_logits
        loss = tf.zeros_like(task_logits[0])
        for i in range(self.task_num - 1):
            for j in range(i+1, self.task_num):
                loss += tf.square(task_logits[i] - task_logits[j])
        loss /= np.math.factorial(self.task_num) / (2 * np.math.factorial(self.task_num - 2))    # M!/(2!(M-2)!)
        return loss

    def _smooth_loss(self):
        _input = tf.concat([self.prev_ob_ph, self.task_ph], axis=1)
        prev_logits = self._build_graph(_input, reuse=True)
        loss = tf.reduce_mean(tf.square(self.logits - prev_logits))     # mse
        return loss

    def _build_graph(self, _input, reuse=True):
        with tf.variable_scope(self.scope, reuse=reuse):
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh, reuse=reuse, scope="fully_connected1")
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh, reuse=reuse, scope="fully_connected2")
            logits = tf.contrib.layers.fully_connected(p_h2, self.module_num, activation_fn=tf.identity, reuse=reuse, scope="fully_connected3")
            logits = tf.contrib.layers.softmax(logits)  # softmax output
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def fix_output_module(self, modules):
        if not isinstance(modules, np.ndarray):
            modules = np.array(modules)
        m = tf.constant(modules.T, dtype=self.logits.dtype)
        ones = tf.ones(shape=(tf.shape(self.ob_ph)[0], self.module_num), dtype=self.logits.dtype)
        self.logits = tf.multiply(ones, m)

    def reset_output(self):
        if self.logits != self.orig_logits:
            self.logits = self.orig_logits
