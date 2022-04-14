'''
This script is heavily based on @openai/baselines.gail.mlp_policy.
'''
from baselines.common.mpi_running_mean_std import RunningMeanStd
import maps.tf_util as U
import tensorflow as tf
import gym

class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, **kwargs):
        name = kwargs['prefix'] + name if 'prefix' in kwargs else name
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._init(**kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, hid_size, num_hid_layers, prefix='', task_ph=None, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self.prefix = prefix

        sequence_length = None

        ob = U.get_placeholder(name='ob', dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter", reuse=tf.AUTO_REUSE):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('pol', reuse=tf.AUTO_REUSE):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            if task_ph is not None:     # UsInG a TeNsOr As A pYtHoN `bOoL` iS nOt AlLoWeD.
                obz = tf.concat([obz, task_ph], axis=1)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
        self.features = last_out

    # @property
    # def features(self, stochastic, ob):
    #    return self._features

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
