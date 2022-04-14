from maps.mujoco_dset import Mujoco_Dset, Dset
from maps import logger
import numpy as np


def read_npz(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, (list, dict, tuple)):
        res = data
    elif isinstance(data, np.lib.npyio.NpzFile):
        res = dict(data)
        data.close()
    else:
        print("incompatible type of data to unzip...")
        res = None
    return res


class ModularDset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True, seed=0, load=True, load_env=None):
        """
        Modular dataset that holds the datasets of all envs separately
        expert path: the path where the dataset file is
        train_fraction: train-validation split
        traj_limitation: number of trajectories to be loaded
        randomize: randomise the state-action pairs' order
        seed: randomisation seed
        load: load the entire datasets or just the env ids (for eval)
        load_env: optional env name to load the dataset of only one env (for eval)
        """
        traj_data = read_npz(expert_path)
        if not traj_data:
            print("no dataset to load. Terminating...")
            exit(1)
        for key in traj_data:
            traj_data[key] = traj_data[key].item()

        if load_env:
            assert load_env in traj_data
            tmp = {load_env: traj_data[load_env]}
            traj_data = tmp

        self._env_ids = {key: traj_data[key].pop("id") for key in traj_data}
        self.datasets = {key: Dataset(traj_data[key], train_fraction, traj_limitation, randomize, seed, env=key)
                         for key in traj_data} if load else None

    def envs(self):
        return list(self._env_ids)

    def __getitem__(self, env):
        return self.datasets[env]

    def id(self, env):
        return self._env_ids[env]

    def available_batch(self, batch_size, split=None):
        for dataset in self.datasets:
            if not self.datasets[dataset].available_batch(batch_size, split):
                return False
        return True


class DsetExt(Dset):
    def __init__(self, inputs, labels, prev_inputs, randomize, randgen=None):
        self.randgen = randgen
        self.inputs = inputs
        self.prev_inputs = prev_inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels) == len(self.prev_inputs)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            if self.randgen:
                self.randgen.shuffle(idx)
            else:
                np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.prev_inputs = self.prev_inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch_w_prev(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        prev_inputs = self.prev_inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels, prev_inputs

    def available_batch(self, batch_size, split=None):
        '''
        check if batch size is available in current epoch
        '''
        return batch_size < 0 or self.pointer + batch_size < self.num_pairs


class Dataset(Mujoco_Dset):
    """
    Same as Mujoco_Dset, only it doesn't need to load the trajectories
    + save prev state at every timestep
    """
    def __init__(self, data_dict, train_fraction=0.7, traj_limitation=-1, randomize=True, seed=0, env=None):
        if env:
            logger.log(env)

        self._initialise(traj_data=data_dict, train_fraction=train_fraction, traj_limitation=traj_limitation,
                         randomize=randomize, seed=seed)

    def _initialise(self, traj_data, train_fraction, traj_limitation, randomize, seed=0):
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])

        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]
        prev_obs = traj_data['obs'][:traj_limitation].tolist()
        manual_convert = False
        if not isinstance(prev_obs[0], list):
            # for stupid numpy 1.18
            manual_convert = True
            prev_obs = [traj.tolist() for traj in prev_obs]
        assert isinstance(prev_obs[0], list)

        for i in range(len(prev_obs)):
            del prev_obs[i][-1]
            prev_obs[i].insert(0, prev_obs[i][0])

        if manual_convert:
            prev_obs = [np.asarray(traj) for traj in prev_obs]
        prev_obs = np.array(prev_obs)

        for i in range(len(obs)):    # debug test
            for j in range(1, len(obs[i])):
                assert all(obs[i][j-1] == prev_obs[i][j])
        self._randgen = np.random.RandomState(seed)

        def flatten(x):
            # x.shape = (E,), or (E, L, D)
            _, size = x[0].shape
            episode_length = [len(i) for i in x]
            y = np.zeros((sum(episode_length), size))
            start_idx = 0
            for l, x_i in zip(episode_length, x):
                y[start_idx:(start_idx+l)] = x_i
                start_idx += l
            return y
        self.obs = np.array(flatten(obs))
        self.prev_obs = np.array(flatten(prev_obs))
        self.acs = np.array(flatten(acs))
        self.rets = traj_data['ep_rets'][:traj_limitation]
        self.avg_ret = sum(self.rets)/len(self.rets)
        self.std_ret = np.std(np.array(self.rets))
        if self.acs.ndim > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_traj = min(traj_limitation, len(traj_data['obs']))
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = DsetExt(self.obs, self.acs, self.prev_obs, self.randomize, randgen=self._randgen)
        # for behavior cloning
        self.train_set = DsetExt(self.obs[:int(self.num_transition*train_fraction), :],
                                 self.acs[:int(self.num_transition*train_fraction), :],
                                 self.prev_obs[:int(self.num_transition*train_fraction), :],
                                 self.randomize, randgen=self._randgen)
        self.val_set = DsetExt(self.obs[int(self.num_transition*train_fraction):, :],
                               self.acs[int(self.num_transition*train_fraction):, :],
                               self.prev_obs[int(self.num_transition*train_fraction):, :],
                               self.randomize, randgen=self._randgen)
        self.log_info()

    def get_next_batch_w_prev(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch_w_prev(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch_w_prev(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch_w_prev(batch_size)
        else:
            raise NotImplementedError

    def available_batch(self, batch_size, split=None):
        '''
        check if batch size is available in current epoch
        '''
        if split is None:
            return self.dset.available_batch(batch_size)
        elif split == 'train':
            return self.train_set.available_batch(batch_size)
        elif split == 'val':
            return self.val_set.available_batch(batch_size)
        else:
            raise NotImplementedError
