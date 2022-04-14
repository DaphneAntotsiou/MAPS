from baselines.common import set_global_seeds
import os
from maps.tf_util import file_writer
import numpy as np
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
from pathlib import Path
import csv
from maps.statistics import stats


def session_config(num_cpu=None):
    config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        # device_count={'GPU': 0}
    )
    config.gpu_options.allow_growth = True


def csv_2_columnwise_dict(file):
    from collections import defaultdict
    from csv import DictReader

    columnwise_table = defaultdict(list)
    with open(file, 'rU') as f:
        reader = DictReader(f)
        for row in reader:
            for col, dat in row.items():
                columnwise_table[col].append(atof(dat))
    return columnwise_table


def atoi(text):
    return int(text) if text.replace('-', '', 1).isdigit() else text


def atof(text):
    return float(text) if text.replace('.', '', 1).replace('-', '', 1).isdigit() else text


def natural_keys(text):
    import re
    res = [atoi(c) for c in re.split('(\d+)', text) if c.isdigit()]
    return res


def evaluate_hours(func, envs, func_list=None, func_dict=None, seed=0, keyword='hour', step=1,
                   stochastic=True, continue_hour=True, dir_separation=True, dir_name=None):
    """
    :param func: function that runs the evaluation
    :param func_list: list parameters of func
    :param func_dict: dictionary parameters of func
    :param seed: seed of run
    :param keyword: keyword that will be excluded from the name search
    :param step: step between the runs
    :param stochastic: boolean for stochastic or deterministic policy run
    :param continue_hour: continue from the last evaluated step, if a previous run is available
    :param dir_separation: boolean if the checkpoints are separated in different subdirs w/ checkpoint files or
    :param dir_name: optional name of subfolder where evaluation is saved
    :return:
    """
    assert isinstance(envs, dict)

    if 'load_model_path' in func_dict and 'number_trajs' in func_dict:
        checkpoint_path = func_dict['load_model_path']
        trajectories = func_dict['number_trajs']
    else:
        print('No path to evaluate...')
        return

    dir_name = os.path.join(checkpoint_path, dir_name) if dir_name else checkpoint_path

    network_prefix = func_dict['network_prefix'] if 'network_prefix' in func_dict else ''

    hour_path = os.path.join(checkpoint_path, keyword)
    hour_path = hour_path.replace("\\", "/")

    success_stats = None

    paths = []
    if dir_separation:
        from glob import glob
        paths = glob(hour_path + "*")
        paths.sort(key=natural_keys)
    else:
        def get_step(elem):
            return Path(elem).stem.rpartition("_")[2]

        def step_int(elem):
            return int(get_step(elem))

        paths = sorted([os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f))
                        and f.find(".index") != -1 and get_step(f).isnumeric()], key=step_int)
    if not paths:
        print("no checkpoint path found at", checkpoint_path)
        return None

    # add writer for log
    policy_type = "stochastic" if stochastic else "deterministic"
    logdir = {env: os.path.join(dir_name, env, env + "_eval_" + str(func_dict["number_trajs"]), "seed_" + str(seed),
                          policy_type) for env in envs}
    writer = {env: file_writer(logdir[env]) for env in envs}

    # NOTE: BC is the same as hour000

    res_file = {env: logdir[env] + '/res_' + network_prefix + '_' + str(trajectories) + '.csv' for env in envs}

    # find out if it should append or not
    start_hour = {env: 0 for env in envs}  # starting hour to compute
    for env in envs:
        if continue_hour and os.path.isfile(res_file[env]):
            existing_res = csv_2_columnwise_dict(res_file[env])
            if existing_res["step"]:
                start_hour[env] = max(existing_res["step"]) + 1

    mode = {env: 'w' if not start_hour[env] or not os.path.isfile(res_file[env]) else 'a' for env in envs}

    func_dict["find_checkpoint"] = True if dir_separation else False

    is_first = True
    for path in paths:
        path = path.replace("\\", "/")

        if path.find(hour_path) != -1 and dir_separation:
            hour = path.replace(hour_path, '')
            if hour.isdigit():
                hour = int(hour)
            else:
                # is not hour - probably eval
                continue
        elif not dir_separation:
            hour = int(Path(path).stem.rpartition("_")[2])  # get the step number
            path = os.path.splitext(path)[0]    # remove extension
        else:
            # no hour or BC in paths
            continue

        load_model = False
        for env in envs:
            if not (hour < start_hour[env] or (hour % step and hour != 155)):  # add 155 for InvertedPendulum run
                load_model = True
                break
        if not load_model:
            continue

        func_dict['load_model_path'] = path
        name_list = {}
        for env in envs:
            if hour < start_hour[env] or (hour % step and hour != 155):  # add 155 for InvertedPendulum run
                continue
            print("evaluating env", env, " for hour ", hour)

            # reset seed so all runs have the same conditions
            set_global_seeds(seed)  # reset seed because of dataset randomisation
            envs[env]['env'].seed(seed)

            func_dict['envs'] = {env: envs[env]}   # one env at a time, easier to handle

            res, pi = func(*func_list, **func_dict)
            if not res or not pi:
                print("Failed to run " + path + " for env " + env)
                continue
            res = res[env]
            func_dict['d_policy'] = pi
            func_dict['load_model_path'] = None

            # log iteration
            curr_checkpoint_file = tf.train.latest_checkpoint(path) if dir_separation else path
            if curr_checkpoint_file.rfind("iter_") != -1 and \
                    curr_checkpoint_file[curr_checkpoint_file.rfind("iter_") + len("iter_"):].isdigit():
                curr_iter = int(curr_checkpoint_file[curr_checkpoint_file.rfind("iter_") + len("iter_"):])
            elif not dir_separation:
                curr_iter = hour
            else:
                curr_iter = None

            if env not in name_list:
                # get name titles of statistics
                name_list[env] = ['step']

                if curr_iter is not None:
                    name_list[env].append("iteration")

                if mode[env] == 'w':
                    if res and isinstance(res, dict):
                        name_list[env].extend([v for v in res])

                    # write the titles
                    with open(res_file[env], mode=mode[env], newline='') as hour_file:
                        csv_writer = csv.writer(hour_file)
                        csv_writer.writerow(name_list[env])
                    mode[env] = 'a'
                else:
                    with open(res_file[env], newline='') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            header = list(row)
                            break
                        if len(header) > len(name_list[env]):
                            name_list[env].extend(header[len(name_list[env]):])

                if success_stats is None:
                    stat_names = name_list[env]
                    success_stats = stats(stat_names, scope="Evaluation")

            # get results of this hour

            res_list = [hour]

            if curr_iter is not None:
                res_list.append(curr_iter)

            if res and isinstance(res, dict):
                # res_list.extend([(str(v)) for v in res.values() if v is not None])
                stat_list = res_list.copy()
                res_list.extend([(str(res[name])) for name in name_list[env] if name in res and res[name] is not None])
                stat_list.extend([(str(res[name])) for name in stat_names if name in res and res[name] is not None])

            if writer:
                success_stats.add_all_summary(writer[env], [float(v) for v in stat_list], curr_iter if curr_iter is not None else hour)

            with open(res_file[env], mode=mode[env], newline='') as hour_file:
                # save as csv
                csv_writer = csv.writer(hour_file)
                csv_writer.writerow(res_list)

    # has evaluated all the hours
    print("evaluation of hourly results saved at: ", res_file)
    return res_file


def rindex(lst, val, start=None):
    if start is None:
        start = len(lst)-1
    for i in range(start,-1,-1):
        if lst[i] == val:
            return i


def best_checkpoint_dir(log, dir, dir_separation=True, keyword=""):
    """
    Get best checkpoint based on the log from evaluate_hours
    :param log: the csv file with the log, produced by evaluate_hours
    :param dir: optional the directory with all the checkpoints. If None, is log parent
    :param keyword: optional keyword to filter the dir subforlders
    :return: checkpoint file with the best results in log
    """
    max_idx = None
    if os.path.isfile(log):
        log_dict = csv_2_columnwise_dict(log)
        key = 'success' if 'success' in log_dict else 'avg_ret'

        if key not in log_dict:
            print("No appropriate success key in csv. Terminating...")
            exit(1)

        max_idx = rindex(log_dict[key], max(log_dict[key]))
        if 'step' in log_dict:
            step = log_dict['step'][max_idx]
            max_idx = step
        else:
            print("no appropriate step key in csv. Terminating...")
            exit(1)
    else:
        print("log file does not exist. Terminating...")
        exit(1)

    if os.path.isdir(dir):

        if dir_separation:
            paths = [os.path.join(dir, o) for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o)) and o.find(keyword) != -1]
            paths.sort(key=natural_keys)

            for path in paths:
                if log.find(path) != -1:
                    continue    # is log path
                idx = natural_keys(os.path.basename(os.path.normpath(path)))
                if len(idx) and idx[0] == max_idx:
                    return path
        else:
            for f in os.listdir(dir):
                if f.find(".index") != -1:
                    f = f.rstrip(".index")
                    if '_' in f and int(f.rpartition("_")[2]) == max_idx:
                        return f

        # should not reach this
        print("something went wrong... no best path found")
        exit(1)

    print("dir file does not exist. Terminating...")
    exit(1)


def evaluate_sigmas(step, low, high, func, func_list=None, func_dict=None, seed=0, stochastic=True):
    """
    Evaluate a checkpoint for different sigmas
    :param step:
    :param low:
    :param high:
    :param func:
    :param func_list:
    :param func_dict:
    :param seed:
    :param stochastic:
    :return:
    """

    if 'load_model_path' in func_dict and 'number_trajs' in func_dict:
        checkpoint_path = func_dict['load_model_path']
        trajectories = func_dict['number_trajs']
    else:
        print('No path to evaluate...')
        return


    network_prefix = func_dict['network_prefix'] if 'network_prefix' in func_dict else ''

    # add writer for log
    policy_type = "stochastic" if stochastic else "deterministic"
    logdir = os.path.join(checkpoint_path, "sigma_eval_" + str(func_dict["number_trajs"]), "seed_" + str(seed),
                          policy_type)
    writer = file_writer(logdir)
    if writer:
        from il.statistics import stats
    success_stats = None

    res = {}

    import csv
    res_file = logdir + '/res_' + network_prefix + '_' + str(trajectories) + '.csv'

    import tensorflow as tf

    mode = 'w'
    is_first = True
    for sigma in np.arange(low, high + step, step): # closed bounds
        print("evaluating sigma ", sigma)

        # reset seed so all runs have the same conditions
        set_global_seeds(seed)  # reset seed because of dataset randomisation
        if func_list is not None and len(func_list) > 0:
            func_list[0].seed(seed)
            func_list[0].env.env.sigma = sigma
        elif "env" in func_dict:
            func_dict["env"].seed(seed)
            func_dict["env"].env.env.sigma = sigma

        res[sigma] = func(*func_list, **func_dict)

        if is_first:
            # get name titles of statistics
            name_list = ['sigma', 'avg_len', 'avg_ret']

            if res and len(list(res.values())[0]) > 2 and isinstance(list(res.values())[0][2], dict):
                name_list.extend([v for v in list(res.values())[0][2].keys()])

            if writer and success_stats is None:
                success_stats = stats(name_list, scope="Sigma_evaluation")

            if mode == 'w':
                # write the titles
                with open(res_file, mode=mode, newline='') as hour_file:
                    csv_writer = csv.writer(hour_file)
                    csv_writer.writerow(name_list)
                mode = 'a'

        # get results of this sigma

        result = res[sigma]

        avg_len = result[0]
        avg_ret = result[1]
        prc_dict = result[2]

        res_list = [sigma, avg_len, avg_ret]

        if prc_dict and isinstance(prc_dict, dict):
            res_list.extend([(str(v)) for v in prc_dict.values() if v is not None])

        if writer:
            success_stats.add_all_summary(writer, [float(v) for v in res_list], sigma * 1000)   # *1000 because iter int

        with open(res_file, mode=mode, newline='') as sigma_file:
            # save as csv
            csv_writer = csv.writer(sigma_file)
            csv_writer.writerow(res_list)

    # has evaluated all the sigmas
    print("evaluation of sigma results saved at: ", res_file)
    return res_file


def evaluate_runs(func, func_list=None, func_dict=None, seed=0, append_step=-1, run_name="traj"):
    """
    Legacy function for behavior_clone_trajectories
    TODO: replace with evaluate_hours DO WE NEED THIS??
    :param func:
    :param func_list:
    :param func_dict:
    :param seed:
    :param append_step:
    :param run_name:
    :return:
    """
    from baselines.common import set_global_seeds
    from glob import glob
    import os
    from os import listdir
    from os.path import isdir, join

    if 'load_model_path' in func_dict and 'number_trajs' in func_dict:
        checkpoint_path = func_dict['load_model_path']
        trajectories = func_dict['number_trajs']
    else:
        print('No path to evaluate...')
        return

    network_prefix = func_dict['network_prefix'] if 'network_prefix' in func_dict else ''

    def get_step(elem):
        return int(elem.rpartition("_")[2])

    files = sorted([join(checkpoint_path, f) for f in listdir(checkpoint_path) if isdir(join(checkpoint_path, f))
                    and f.find(run_name) != -1], key=get_step)

    if isdir(join(checkpoint_path, "BC")):
        files.insert(0, join(checkpoint_path, "BC"))

    res = {}

    import csv
    res_file = checkpoint_path + '/res_' + network_prefix + '_' + str(trajectories) + '.csv'
    mode = 'w' if append_step < 0 else 'a'

    for file in files:

        step = int(file.rpartition("_")[2])     # get the step number

        if step < append_step:
            continue

        print("evaluating step ", step)

        # reset seed so all runs have the same conditi1ons
        set_global_seeds(seed)  # reset seed because of dataset randomisation
        if func_list is not None and len(func_list) > 0:
            func_list[0].seed(seed)
        elif "env" in func_dict:
            func_dict["env"].seed(seed)

        func_dict['load_model_path'] = file
        res[step] = func(*func_list, **func_dict)

        if mode == 'w':
            # write the titles
            with open(res_file, mode=mode, newline='') as step_file:
                writer = csv.writer(step_file)

                name_list = ['step', 'avg_len', 'avg_ret']

                if res and len(list(res.values())[0]) > 2 and isinstance(list(res.values())[0][2], dict):
                    name_list.extend([v for v in list(res.values())[0][2].keys()])
                writer.writerow(name_list)
            mode = 'a'

        with open(res_file, mode=mode, newline='') as step_file:
            writer = csv.writer(step_file)

            result = res[step]

            avg_len = result[0]
            avg_ret = result[1]
            prc_dict = result[2]

            res_list = [step, avg_len, avg_ret]
            if prc_dict and isinstance(prc_dict, dict):
                res_list.extend([str(v) for v in prc_dict.values() if v])
            writer.writerow(res_list)

    print("evaluation of step results saved at: ", res_file)


def get_module_layers_size_from_checkpoint(fname):
    '''

    :param fname: checkpoint file path
    :return: modules, layers, dictionary hidden size, router hidden size
    '''
    if fname is not None and tf.train.checkpoint_exists(fname):
        reader = pywrap_tensorflow.NewCheckpointReader(fname)
        var_to_shape_map = reader.get_variable_to_shape_map()
        modules = 0
        layers = 0
        d_size = 0
        r_size = 0
        mod_key = "module"
        layer_key = "fc"
        router_key = "fully_connected1"
        for tensor in var_to_shape_map:
            if tensor.find(mod_key) != -1:
                curr_mod = int(tensor[tensor.find(mod_key) + len(mod_key): tensor.find("/", tensor.find(mod_key) + len(mod_key))])
                modules = max(modules, curr_mod)
            if tensor.find(layer_key) != -1:
                curr_layer = int(tensor[tensor.find(layer_key) + len(layer_key): tensor.find("/",
                                                                                             tensor.find(layer_key) +
                                                                                             len(layer_key))])
                layers = max(layers, curr_layer)
                if not d_size and len(var_to_shape_map[tensor]) == 1 and \
                    'vf/fc1/bias' in tensor:
                    d_size = var_to_shape_map[tensor][0]

            if not r_size and tensor.find(router_key) != -1 and len(var_to_shape_map[tensor]) == 1 and \
                    'bias' in tensor:
                r_size = var_to_shape_map[tensor][0]
                if not d_size and r_size:
                    d_size = r_size
                elif not r_size and d_size:
                    r_size = d_size
        return modules+1, layers, d_size, r_size

