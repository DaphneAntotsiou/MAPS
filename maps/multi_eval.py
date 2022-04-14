import argparse
import maps.tf_util as U
from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines import bench
import tensorflow as tf
import gym
from maps.dataset import ModularDset
import os.path as osp

from maps.modular_policy import ModularPolicy
from maps.selector_policy_smooth import SelectorPolicy
from tqdm import tqdm
import numpy as np
import os
from copy import deepcopy
import gym_extensions.continuous.mujoco
import sawyer.v2
from maps.env_evaluation.Success import success
from maps.eval_utils import get_module_layers_size_from_checkpoint
from maps.evaluate import evaluate
from multiprocessing import Pool


def none_or_str(value):
    if value == 'None':
        return None
    return value


def argsparser():
    parser = argparse.ArgumentParser("Modular Adaptive Policy Selection for Multi-Task Imitation evaluation")
    parser.add_argument('--dataset', help='dataset directory of all the environments', default="data/tasks_10_traj_20_onehot.npz")
    parser.add_argument('--env_id', help='environment ID', type=none_or_str, default=None)
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='push-v2')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default="reach-v2")
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='pick-place-v2')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='door-open-v2')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='drawer-open-v2')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='drawer-close-v2')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='window-close-v2')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='window-open-v2')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='button-press-topdown-v2')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='peg-insert-side-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=350)
    parser.add_argument('--log_dir', help='the directory to save log file', default=None)
    parser.add_argument('--load_model_path', help='model directory', default="checkpoint/tasks_10_traj_20_short_onehot-5_15_3layer_32bs_1.0_0.1_0.5_1.0_0.75/1/")
    boolean_flag(parser, 'eval_hours', default=True, help='evaluate all the checkpoints of the train run')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'render', default=False, help='render trajectories for qualitative evaluation')

    return parser.parse_args()


def main(args):
    set_global_seeds(args.seed)
    dataset = ModularDset(args.dataset, seed=args.seed, load=False)
    if not args.env_id:
        # create all the environments
        envs = [key for key in dataset.envs()]
        envs_id = [dataset.id(key) for key in envs]
    else:
        assert args.env_id in dataset.envs()
        envs = [args.env_id]
        envs_id = [dataset.id(key) for key in envs]

    cpus = len(envs)
    pool = Pool(processes=cpus)

    try:
        for env, env_id in zip(envs, envs_id):
            # pool.apply_async(eval_per_env, args=(args, env, env_id, 100, False))
            eval_per_env(args, env, env_id, number_trajs=100, find_checkpoint=False)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()


def eval_per_env(args, env, env_id, number_trajs=100, find_checkpoint=True):
    graph = tf.Graph()
    sess = U.make_session(graph=graph)
    with graph.as_default():
        with sess.as_default():
            U.ALREADY_INITIALIZED.clear()

            envs = {env: gym.make(env)}

            envs = {env: {'env': bench.Monitor(envs[key], None, allow_early_resets=True),
                          'taskid': env_id} for key in envs}
            set_global_seeds(args.seed)

            for key in envs:
                envs[key]['env'].seed(args.seed)

            func_list = []
            func_dict = {
                'load_model_path': args.load_model_path,
                'timesteps_per_batch': 1024,
                'number_trajs': number_trajs,
                'stochastic_policy': args.stochastic_policy,
                'save': args.save_sample,
                'render': args.render,
                'find_checkpoint': find_checkpoint
            }

            if args.eval_hours:
                from maps.eval_utils import evaluate_hours
                func_dict["find_checkpoint"] = False
                evaluate_hours(runner, envs, func_list=func_list, func_dict=func_dict, seed=args.seed,
                               keyword=os.path.basename(os.path.normpath(args.load_model_path)),
                               dir_separation=False,
                               dir_name="multithread")
            else:
                func_dict["save_module_dist"] = True
                env_dict, _ = runner(envs, *func_list, **func_dict)

            for key in envs:
                envs[key]['env'].close()
            return True


def init_model(envs, dir, find_checkpoint=True, d_policy=None):
    checkpoint_file = tf.train.latest_checkpoint(dir) if find_checkpoint else dir
    if not d_policy:
        taskid_space = None
        if isinstance(list(envs.values())[0]['taskid'], int):
            from gym.spaces.box import Box
            taskid_space = Box(low=0, high=len(envs), shape=(1,), dtype=np.uint8)
        elif isinstance(list(envs.values())[0]['taskid'], np.ndarray):
            # is onehot
            from gym.spaces.box import Box
            taskid_space = Box(low=0, high=1, shape=list(envs.values())[0]['taskid'].shape, dtype=np.uint8)
        else:
            print(envs)
            raise NotImplementedError

        module_num, layers, d_size, r_size = get_module_layers_size_from_checkpoint(checkpoint_file)
        gaussian = bool(checkpoint_file.find("no_gaussian") == -1)
        tmp = str(tf.get_default_session())
        # tmp2 = str(tf.get_default_graph())
        print("env " + list(envs.keys())[-1] + " module_num " + str(module_num) + " layers " + str(layers) +
              " d_size " + str(d_size) + " r_size " + str(r_size) + " creating router policy in graph " +
              str(tf.get_default_graph()) + " and session " + tmp)
        routing_policy = SelectorPolicy("router", module_num=module_num, task_space=taskid_space,
                                        ob_space=list(envs.values())[0]['env'].observation_space,
                                        hidden_size=r_size, loss=False)
        print("creating dictionary policy " + str(tf.get_default_graph()), " and session ", str(tf.get_default_session()))
        d_policy = ModularPolicy("d_pi", selector_policy=routing_policy, module_num=module_num,
                                 ob_space=list(envs.values())[0]['env'].observation_space,
                                 ac_space=list(envs.values())[0]['env'].action_space,
                                 hid_size=d_size, num_hid_layers=layers,
                                 gaussian_fixed_var=gaussian)

    print("loading checkpoint tensors")
    U.ALREADY_INITIALIZED.clear()
    is_vaild = True  # checkpoint is valid
    if not U.load_checkpoint_variables(checkpoint_file):
        is_vaild = False

    if not is_vaild:
        # invalid checkpoint
        print("Invalid checkpoint at " + dir + " for envs", envs.keys())
        return None
    U.initialize()
    return d_policy


def runner(envs, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, render=True,
           find_checkpoint=True, d_policy=None, save_module_dist=False):
    if load_model_path:
        d_policy = init_model(envs, load_model_path, find_checkpoint=find_checkpoint, d_policy=d_policy)
        if d_policy is None:
            return None, None
    env_dict, trajectories = evaluate(envs, d_policy, timesteps_per_batch, number_trajs, stochastic_policy,
                                      render, save=save, module_dist=save_module_dist)
    for env in env_dict:
        res_dict = env_dict[env]
        for key in res_dict:
            if res_dict[key] is not None and "module" not in key:
                print(key + " policy mean = " + str(res_dict[key]))

        if save:
            filename = osp.join(os.path.dirname(load_model_path), 'policy_traj_' + str(number_trajs) + "_" + env)
            np.savez(filename, **trajectories[env])
            print("saved at", filename)

        if save_module_dist:
            import csv

            checkpoint_file = tf.train.latest_checkpoint(load_model_path) if find_checkpoint else load_model_path
            head, tail = os.path.split(checkpoint_file)
            head = os.path.join(head, "multithread")
            os.makedirs(head, exist_ok=True)
            checkpoint_file = os.path.join(head, tail)
            filename = checkpoint_file + '_' + env + "_module_dist" + str(number_trajs) + ".csv"
            with open(filename, mode='w') as module_file:
                file_writer = csv.DictWriter(module_file, env_dict[env].keys())
                file_writer.writeheader()
                file_writer.writerow(env_dict[env])

    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    return env_dict, d_policy


if __name__ == '__main__':
    args = argsparser()
    main(args)
