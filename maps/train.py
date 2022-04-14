'''
This script's BC is based on @openai/baselines.gail.behavior_clone
'''
import argparse
import maps.tf_util as U
from baselines.common import set_global_seeds
from baselines.common.misc_util import boolean_flag
from baselines import bench
import tensorflow as tf
import gym
from maps.dataset import ModularDset
import os.path as osp
from maps import logger
import logging
from mpi4py import MPI
from maps.modular_policy import ModularPolicy
from maps.selector_policy_smooth import SelectorPolicy
from baselines.common.mpi_adam import MpiAdam
from maps.statistics import stats
from tqdm import tqdm
import numpy as np
import os
try:
    import gym_extensions.continuous.mujoco
    import sawyer.v2
except ImportError:
    pass


def none_or_str(value):
    if value == 'None':
        return None
    return value


def get_task_name(dataset_path, sharing, exploration, sparsity, task_w, module_num, batch_size, layers, traj_limitation,\
                  smooth):
    without_extra_slash = os.path.normpath(dataset_path)
    last_part = os.path.basename(without_extra_slash)
    last_part = last_part[:-4]   # remove .npz
    traj_limitation = "" if traj_limitation < 0 else "-" + str(traj_limitation)
    name = last_part + traj_limitation + "_" + str(module_num) + "_" + str(layers) + "layer_" + str(batch_size) + "bs_" \
           + str(sharing) + "_" + str(exploration) + "_" + str(sparsity) + "_" + str(smooth) + "_" + str(task_w)
    return name


def argsparser():
    parser = argparse.ArgumentParser("Modular Adaptive Policy Selection for Multi-Task Imitation training")
    parser.add_argument('--dataset', help='dataset directory of all the environments', default="data/tasks_10_traj_20_onehot.npz")
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--policy_hidden_size', type=int, default=128)
    parser.add_argument('--module_num', type=int, default=10)
    parser.add_argument('--layers', help="number of module hidden layers", type=int, default=3)
    parser.add_argument('--log_dir', help='the directory to save log file', default=None)
    parser.add_argument('--checkpoint_dir', help='checkpoint directory', default="checkpoint/")
    parser.add_argument('--timesteps_per_batch', help='total generator timesteps per batch', type=int, default=32)
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=0)
    parser.add_argument('--num_iters', help='max number of iterations', type=int, default=0)
    parser.add_argument('--r_lr', type=float, help='selector learning rate', default=3e-4)
    parser.add_argument('--d_lr', type=float, help='module learning rate', default=3e-4)
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--exploration', type=float, default=0.1)
    parser.add_argument('--sharing', type=float, default=1.0)
    parser.add_argument('--smooth', type=float, default=1.0)
    parser.add_argument('--task_w', type=float, default=0.75)
    parser.add_argument('--traj_limitation', type=int, default=15)
    parser.add_argument('--continue_dir', type=none_or_str, default=None)
    boolean_flag(parser, 'multi_eval', default=True, help='run evaluation at the end of training')
    boolean_flag(parser, 'eval_hours', default=True, help='evaluate all the checkpoints of the train run')
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy')

    return parser.parse_args()


def main(args):
    sess = U.make_session().__enter__()
    args.seed += 10000 * MPI.COMM_WORLD.Get_rank()

    set_global_seeds(args.seed)

    if args.continue_dir is None:
        task_name = get_task_name(args.dataset, args.sharing, args.exploration, args.sparsity, args.task_w, args.module_num,
                                  args.timesteps_per_batch, args.layers, args.traj_limitation, args.smooth)
        args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name, str(args.seed))
    else:
        from maps.evaluate import get_module_layers_size_from_checkpoint
        args.checkpoint_dir = args.continue_dir + "_continue"
        # args.continue_dir = tf.train.latest_checkpoint(args.continue_dir)
        if not args.continue_dir:
            print("No valid checkpoint to continue in dir", args.continue_dir)
            exit(1)
        args.module_num, args.layers, args.policy_hidden_size, _ = \
            get_module_layers_size_from_checkpoint(args.continue_dir)
        task_name = get_task_name(args.dataset, args.sharing, args.exploration, args.sparsity, args.task_w, args.module_num,
                                  args.timesteps_per_batch, args.layers, args.traj_limitation, args.smooth)

    args.log_dir = osp.join(args.log_dir, task_name) if args.log_dir and not args.continue_dir else \
        osp.join(args.checkpoint_dir, 'log')

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(dir=args.log_dir, append=(args.continue_dir is not None))
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    dataset = ModularDset(args.dataset, seed=args.seed, traj_limitation=args.traj_limitation)
    # create all the environments
    envs = {key: gym.make(key) for key in dataset.envs()}

    envs = {key: {'env': bench.Monitor(envs[key], None, allow_early_resets=True),
                  'taskid': dataset.id(key)} for key in envs}
    args.timesteps_per_batch *= len(envs.keys())    # multiply by number of envs

    for key in envs:
        envs[key]['env'].seed(args.seed)
    gym.logger.setLevel(logging.WARN)

    taskid_space = None
    if isinstance(list(envs.values())[0]['taskid'], int):
        from gym.spaces.box import Box
        taskid_space = Box(low=0, high=len(envs), shape=(1,), dtype=np.uint8)
    elif isinstance(list(envs.values())[0]['taskid'], np.ndarray):
        # is onehot
        from gym.spaces.box import Box
        taskid_space = Box(low=0, high=1, shape=list(envs.values())[0]['taskid'].shape, dtype=np.uint8)
    else:
        raise NotImplementedError
    logger.log("sparsity " + str(args.sparsity))
    logger.log("exploration " + str(args.exploration))
    logger.log("sharing " + str(args.sharing))
    logger.log("smoothing " + str(args.smooth))
    logger.log("task_w " + str(args.task_w))

    selector_policy = SelectorPolicy("router", module_num=args.module_num, task_num=len(envs), task_space=taskid_space,
                                     ob_space=list(envs.values())[0]['env'].observation_space,
                                     hidden_size=args.policy_hidden_size, c_exploration=args.exploration,
                                     c_sharing=args.sharing, c_sparsity=args.sparsity, c_smooth=args.smooth, loss=True)
    module_policy = ModularPolicy("d_pi", selector_policy=selector_policy, module_num=args.module_num,
                                  ob_space=list(envs.values())[0]['env'].observation_space,
                                  ac_space=list(envs.values())[0]['env'].action_space,
                                  hid_size=args.policy_hidden_size, num_hid_layers=args.layers)

    for key in envs:
        envs[key]['env'].close()   # don't actually need them for BC

    train(envs,
          args.seed,
          selector_policy,
          module_policy,
          dataset,
          args.num_iters,
          args.timesteps_per_batch,
          args.save_per_iter,
          args.checkpoint_dir,
          args.log_dir,
          args.r_lr,
          args.d_lr,
          c_task=args.task_w,
          continue_checkpoint=args.continue_dir
          )


def train(envs,
          seed,
          selector_policy,
          module_policy,
          dataset,
          num_iters,
          timesteps_per_batch,
          save_per_iter,
          checkpoint_dir,
          log_dir,
          r_lr=None,
          d_lr=None,
          adam_epsilon=1e-5,
          continue_checkpoint=None,
          c_task=0.75,
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    # divide the timesteps to the threads
    timesteps_per_batch = -(-timesteps_per_batch // nworkers)       # get ceil of division

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    def allmean_list(x):
        assert isinstance(x, list)
        for i in range(len(x)):
            x[i] = allmean(np.asarray(x[i]))
        return x

    val_per_iter = max(min(int(num_iters/10), 1000), 1)

    ob = U.get_placeholder_cached("ob")
    ac = module_policy.action_sample_placeholder([None])
    stochastic = U.get_placeholder_cached("stochastic")

    bc_loss = tf.reduce_mean(tf.square(ac - module_policy.ac))
    loss = c_task * bc_loss + (1 - c_task) * selector_policy.total_loss

    pi_var_list = module_policy.get_trainable_variables()
    selector_var_list = selector_policy.get_trainable_variables()
    var_list = pi_var_list + selector_var_list

    pi_lossandgrad = U.function([ob, selector_policy.task_ph, selector_policy.prev_ob_ph, ac, stochastic], [bc_loss, loss] + [U.flatgrad(loss, pi_var_list)])
    selector_lossandgrad = U.function([ob, selector_policy.task_ph, selector_policy.prev_ob_ph, ac, stochastic],
                                    selector_policy.losses + [loss] + [U.flatgrad(loss, selector_var_list)])

    pi_loss = U.function([ob, selector_policy.task_ph, selector_policy.prev_ob_ph, ac, stochastic], [bc_loss, loss])

    pi_adam = MpiAdam(pi_var_list, epsilon=adam_epsilon)
    selector_epsilon = adam_epsilon
    selector_adam = MpiAdam(selector_var_list, epsilon=selector_epsilon)

    if rank == 0 and log_dir is not None:
        train_stats = stats(["Train_Loss", "BC_Train_Loss"] + selector_policy.loss_name, scope="Train_Loss")
        val_stats = stats(["Total_Evaluation_Loss", "BC_Eval_Loss"], scope="Validation_Loss")
        eval_stats = None
        writer = U.file_writer(log_dir)

    # load previous checkpoint if specified
    if continue_checkpoint and not U.load_checkpoint_variables(continue_checkpoint):
            print(" failed to load checkpoint", continue_checkpoint)
            exit(1)

    start_iter = [int(s) for s in continue_checkpoint.split("/") if s.isdigit()][-1] if continue_checkpoint else 0

    U.initialize()

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)

    pi_adam.sync()
    selector_adam.sync()

    prev_val_loss = 1e10
    logger.log("training MAPS with Behavioural Cloning...")
    task_batch = timesteps_per_batch // len(envs.keys())
    done = False
    for iter_so_far in tqdm(range(start_iter, np.uint64(num_iters + 1))):

        ob_expert, ac_expert, task_expert, prev_obs = get_next_batch(dataset, task_batch, 'train')
        # logits = selector_policy.output(ob_expert, task_expert)

        *selector_train_loss, r_g = selector_lossandgrad(ob_expert, task_expert, prev_obs, ac_expert, True)

        r_g = allmean(r_g)
        selector_adam.update(r_g, r_lr)

        bc_train_loss, pi_train_loss, pi_g = pi_lossandgrad(ob_expert, task_expert, prev_obs, ac_expert, True)

        pi_g = allmean(pi_g)
        pi_adam.update(pi_g, d_lr)

        del selector_train_loss[-1]

        # get mean losses of all threads
        selector_train_loss = allmean_list(selector_train_loss)
        bc_train_loss, pi_train_loss = allmean_list([bc_train_loss, pi_train_loss])

        if rank == 0 and log_dir is not None:
            train_stats.add_all_summary(writer, [pi_train_loss, bc_train_loss] + selector_train_loss, iter_so_far)
            if not iter_so_far % val_per_iter:
                logger.record_tabular("IterationsSoFar", iter_so_far)
                logger.record_tabular("TrainLoss", pi_train_loss)
                logger.record_tabular("BCTrainLoss", bc_train_loss)
                for i in range(len(selector_train_loss)):
                    logger.record_tabular(selector_policy.loss_name[i], selector_train_loss[i])

        if iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert, task_expert, prev_obs = get_next_batch(dataset, task_batch, 'val')
            bc_val_loss, val_loss = pi_loss(ob_expert, task_expert, prev_obs, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(pi_train_loss, val_loss))
            logger.log("BC Training loss: {}, BC Validation loss: {}".format(bc_train_loss, bc_val_loss))
            loss_diff = val_loss - prev_val_loss
            prev_val_loss = val_loss
            logger.log("Validation loss reduced by ", str(loss_diff))
            if rank == 0 and log_dir is not None:
                val_stats.add_all_summary(writer, [val_loss, bc_val_loss], iter_so_far)
                logger.record_tabular("ValidationLoss", val_loss)
                logger.record_tabular("ValidationBCLoss", bc_val_loss)

            # if abs(loss_diff) < 0.0005 or loss_diff > 0.0005:
            #     done = True
            #     # print("no change in evaluation loss, terminating")
            #     # break
        if rank == 0 and log_dir is not None and not iter_so_far % val_per_iter:
            logger.log("********** Iteration %i ************" % iter_so_far)
            logger.dump_tabular()

        if checkpoint_dir is not None and rank == 0 and iter_so_far % save_per_iter == 0:
            os.makedirs(osp.dirname(checkpoint_dir), exist_ok=True)
            savedir_fname = osp.join(checkpoint_dir, str(iter_so_far).zfill(5))
            U.save_no_prefix(savedir_fname, var_list=None, write_meta_graph=False)
        if done:
            print("no change in evaluation loss, terminating")
            break
    if rank == 0:
        os.makedirs(osp.dirname(checkpoint_dir), exist_ok=True)
        savedir_fname = osp.join(checkpoint_dir, "_final")
        U.save_no_prefix(savedir_fname, var_list=None, write_meta_graph=False)


def get_next_batch(dataset, batch_size, split, shuffle=True):
    """
    get the next batch of concatenated obs, acs and task ids
    :param dataset: the modular dataset
    :param batch_size: batch size for each task
    :return: obs, acs, taskids, prev obs
    """
    ob_expert = None
    ac_expert = None
    task_expert = None
    prev_ob_expert = None
    envs = dataset.envs()
    for env in envs:
        # TODO: find more efficient way to concat task obs and acs
        # concat all tasks
        obs, acs, prev_obs = dataset[env].get_next_batch_w_prev(batch_size, split)
        if ob_expert is None:
            ob_expert = obs
            ac_expert = acs
            prev_ob_expert = prev_obs
            task_expert = np.array([dataset.id(env) for _ in range(np.shape(obs)[0])])
        else:
            ob_expert = np.concatenate([ob_expert, obs])
            ac_expert = np.concatenate([ac_expert, acs])
            prev_ob_expert = np.concatenate([prev_ob_expert, prev_obs])
            task_expert = np.concatenate([task_expert,  np.array([dataset.id(env) for _ in range(np.shape(obs)[0])])])

    if shuffle:
        indices = np.arange(ob_expert.shape[0])
        np.random.shuffle(indices)
        ob_expert = ob_expert[indices]
        ac_expert = ac_expert[indices]
        task_expert = task_expert[indices]
        prev_ob_expert = prev_ob_expert[indices]

    return ob_expert, ac_expert, task_expert, prev_ob_expert


if __name__ == '__main__':
    args = argsparser()
    main(args)
    if args.multi_eval:
        U.ALREADY_INITIALIZED.clear()
        command = "python multi_eval.py --dataset {} --load_model_path {} --{}eval_hours".format(
            args.dataset, args.checkpoint_dir, "" if args.eval_hours else "no-")
        logger.log("running", command)
        os.system(command)

