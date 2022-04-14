import tensorflow as tf
from maps.mlp_policy_ext import MlpPolicy
import maps.tf_util as U
import numpy as np
from baselines.common.distributions import make_pdtype
import gym


class ModularPolicy(object):
    def __init__(self, name, selector_policy, module_num, ob_space, ac_space, **kwargs):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._init(selector_policy, module_num, ob_space, ac_space, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, selector_policy, module_num, ob_space, ac_space, gaussian_fixed_var=True, **kwargs):
        sequence_length = None
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        self.selector_policy = selector_policy
        self._modules = [MlpPolicy("module" + str(i), ob_space=ob_space, **kwargs) for i in range(module_num)]
        f_dic = tf.stack([m.features for m in self._modules], axis=1)  # features dictionary
        features = tf.einsum('ijk,ij->ijk', f_dic, selector_policy.logits)

        # concat all the module features
        feature_list = tf.unstack(features, axis=1)
        features = tf.concat(feature_list, axis=1)

        self.pdtype = pdtype = make_pdtype(ac_space)
        with tf.variable_scope('distribution', reuse=tf.AUTO_REUSE):
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(features, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(features, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        #######

        self._act = U.function([ob, self.selector_policy.task_ph, stochastic], [self.ac, self.selector_policy.logits])
        # self._act = U.function([ob, stochastic], [self.ac, self.selector_policy.logits])

    def act(self, ob, task, stochastic, get_scores=False):
        if not isinstance(task, np.ndarray):
            task = np.array(task)
        ac1, scores = self._act(ob[None], task[None], stochastic)
        # ac1, scores = self._act(ob[None], stochastic)
        return (ac1[0], scores[0]) if get_scores else ac1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    def action_sample_placeholder(self, shape=[None]):
        return self.pdtype.sample_placeholder(shape)
