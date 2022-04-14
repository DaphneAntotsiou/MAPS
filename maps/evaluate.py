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
try:
    import gym_extensions.continuous.mujoco
    import sawyer.v2
except ImportError:
    pass
from maps.env_evaluation.Success import success
from maps.eval_utils import get_module_layers_size_from_checkpoint
import time
from baselines.common import colorize
from contextlib import contextmanager


def none_or_str(value):
    if value == 'None':
        return None
    return value


def argsparser():
    parser = argparse.ArgumentParser("Modular Adaptive Policy Selection for Multi-Task Imitation evaluation")
    parser.add_argument('--dataset', help='dataset directory of all the environments', default="data/tasks_10_traj_20_onehot.npz")
    parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='HalfCheetahGravityOneAndHalf-v0')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='push-v2')
    # parser.add_argument('--env_id', help='environment ID', type=none_or_str, default='reach-v2')
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
    sess = U.make_session().__enter__()

    set_global_seeds(args.seed)

    dataset = ModularDset(args.dataset, seed=args.seed, load=False)
    if not args.env_id:
        # create all the environments
        envs = {key: gym.make(key) for key in dataset.envs()}
    else:
        assert args.env_id in dataset.envs()
        envs = {args.env_id: gym.make(args.env_id)}

    envs = {key: {'env': bench.Monitor(envs[key], None, allow_early_resets=True),
                  'taskid': dataset.id(key)} for key in envs}

    set_global_seeds(args.seed)
    for key in envs:
        envs[key]['env'].seed(args.seed)
    # TODO: check all envs have the same state and action space

    dir_separation = True

    func_list = []
    func_dict = {
                 'load_model_path': args.load_model_path,
                 'timesteps_per_batch': 1024,
                 'number_trajs': 100,
                 'stochastic_policy': args.stochastic_policy,
                 'save': args.save_sample,
                 'render': args.render,
                 'find_checkpoint': False
                }

    if args.eval_hours:
        from maps.eval_utils import evaluate_hours
        func_dict["find_checkpoint"] = False
        evaluate_hours(runner, envs, func_list=func_list, func_dict=func_dict, seed=args.seed,
                       keyword=os.path.basename(os.path.normpath(args.load_model_path)),
                       dir_separation=False,
                       dir_name=None)
    else:
        func_dict["save_module_dist"] = True
        env_dict, _ = runner(envs, *func_list, **func_dict)

    for key in envs:
        envs[key]['env'].close()


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
        print("creating selector policy")
        routing_policy = SelectorPolicy("router", module_num=module_num, task_space=taskid_space,
                                        ob_space=list(envs.values())[0]['env'].observation_space,
                                        hidden_size=r_size, loss=False)
        print("creating modular policy")
        d_policy = ModularPolicy("d_pi", selector_policy=routing_policy, module_num=module_num,
                                 ob_space=list(envs.values())[0]['env'].observation_space,
                                 ac_space=list(envs.values())[0]['env'].action_space,
                                 hid_size=d_size, num_hid_layers=layers)
    print("loading checkpoint tensors")
    U.ALREADY_INITIALIZED.clear()
    is_vaild = True  # checkpoint is valid
    if not U.load_checkpoint_variables(checkpoint_file):
        is_vaild = False

    if not is_vaild:
        # invalid checkpoint
        print("Invalid checkpoint at ", dir)
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
                                      render, save=False, module_dist=save_module_dist)
    for env in env_dict:
        res_dict = env_dict[env]
        for key in res_dict:
            if res_dict[key] is not None and "module" not in key:
                print(key + " policy mean = " + str(res_dict[key]))

        if save:
            filename = osp.join(load_model_path, 'policy_traj_' + str(number_trajs) + "_" + env)
            np.savez(filename, **trajectories[env])
            print("saved at", filename)

        if save_module_dist:
            import csv
            checkpoint_file = tf.train.latest_checkpoint(load_model_path) if find_checkpoint else load_model_path
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


def evaluate(envs, d_policy, timesteps_per_batch, number_trajs, stochastic_policy, render, save=False,
             module_dist=False):
    reward_threshold = -1
    success_only = False
    env_trajectories = {} if save else None
    env_res_dict = {}
    for entry in envs:
        print("evaluating env", entry)

        env = envs[entry]['env']
        taskid = envs[entry]['taskid']

        obs_list = []
        acs_list = []
        len_list = []
        ret_list = []
        trajectories = {}
        res_dict = {}  # return dict

        for _ in tqdm(range(number_trajs), ascii=True, file=os.sys.stdout):
            while True:
                traj, traj_dict = traj_1_generator(d_policy, env, taskid=taskid, horizon=timesteps_per_batch,
                                                   stochastic=stochastic_policy, render=render, module_dist=module_dist)
                if success_only and 'success' in traj and traj['success']:
                    break
                if not success_only and (reward_threshold < 0 or traj['ep_rets'] >= reward_threshold):
                    break
            obs, acs, ep_len, ep_ret = traj['obs'], traj['acs'], traj['lens'], traj['ep_rets']
            for key in traj:
                if key not in trajectories:
                    trajectories[key] = []
                trajectories[key].append(traj[key])
            obs_list.append(obs)
            acs_list.append(acs)

            len_list.append(ep_len)
            ret_list.append(ep_ret)

            for key in traj_dict.keys():
                if key not in res_dict:
                    res_dict[key] = 0
                res_dict[key] += float(traj_dict[key])  # num of successes
        for key in trajectories:
            trajectories[key] = np.asarray(trajectories[key])
        if save:
            # save all the trajectories only if a save is required
            env_trajectories[entry] = trajectories

        for key in res_dict:    # mean success
            res_dict[key] /= number_trajs
        # add reward to res_dict
        res_dict['reward'] = sum(ret_list) / len(ret_list)
        if 'success' not in res_dict:
            res_dict['success'] = success(entry, res_dict['reward'])

        env_res_dict[entry] = res_dict

    return env_res_dict, env_trajectories


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, taskid, horizon, stochastic, render, module_dist=False):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []
    qpos = []
    qvel = []
    traj = {}
    score_dist = None

    res_keys = ['success']
    res_dict = {}

    ob = env.reset()

    if render:
        env.render()

    while True:
        if module_dist:
            ac, score = pi.act(ob, taskid, stochastic, get_scores=True)
            if score_dist is None:
                score_dist = score
            else:
                score_dist += score
        else:
            ac = pi.act(ob, taskid, stochastic, get_scores=False)

        obs.append(ob)
        news.append(new)
        acs.append(ac)

        if hasattr(env.env.env, 'gs'):
            state = env.env.env.gs()
            for key in state:
                if key not in traj:
                    traj[key] = []
                traj[key].append(state[key])
        else:
            qpos.append(deepcopy(env.env.env.data.qpos))
            qvel.append(deepcopy(env.env.env.data.qvel))

        ob, rew, new, frame_dict = env.step(ac)

        for key in res_keys:
            if key == "success" and hasattr(env.env.env, 'success'):
                res_dict[key] = env.env.env.success
            elif key in frame_dict.keys():
                res_dict[key] = frame_dict[key]

        if render:
            env.render()

        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1

        if new or t >= horizon:
            break
        t += 1

    if score_dist is not None:
        score_dist /= cur_ep_len
        for i in range(len(score_dist)):
            res_dict["module"+str(i+1)] = score_dist[i]

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)

    traj.update({"obs": obs, "rews": rews, "new": news, "acs": acs,
                 "ep_rets": cur_ep_ret, "lens": cur_ep_len})
    if hasattr(env.env.env, 'gs'):
        state = env.env.env.gs()
        for key in state:
            traj[key] = np.asarray(traj[key])
    else:
        traj.update({"qpos": qpos, "qvel": qvel})

    for key in res_keys:
        if key in res_dict.keys():
            traj[key] = res_dict[key]
    return traj, res_dict


if __name__ == '__main__':
    args = argsparser()
    main(args)
