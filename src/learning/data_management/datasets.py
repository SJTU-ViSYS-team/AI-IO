from abc import ABC, abstractmethod
import os
import random

import h5py
import numpy as np
from torch.utils.data import Dataset

import learning.utils.pose as pose


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """

    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass


class ModelSequence(CompiledSequence):
    def __init__(self, seq_path, args, **kwargs):
        super().__init__(**kwargs)
        (
            self.ts,
            self.features,
            self.targets,
            self.gyro_raw,
            self.accel_raw,
            self.feat,
            self.traj_target
        ) = (None, None, None, None, None, None, None)

        self.mode = kwargs.get("mode", "train")
        self.perturb_accel = args.perturb_accel
        self.perturb_accel_range = args.perturb_accel_range

        if seq_path is not None:
            self.load(seq_path)

    def load(self, data_path):
        with h5py.File(os.path.join(data_path, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"])
            gyro_raw = np.copy(f["gyro_raw"])
            gyro_calib = np.copy(f["gyro_calib"])
            accel_raw = np.copy(f["accel_raw"])
            accel_calib = np.copy(f["accel_calib"])
            traj_target = np.copy(f["traj_target"])
            rotor_spd = np.copy(f["rotor_spd"])

        self.ts = ts
        self.gyro_raw = gyro_raw
        self.accel_raw = accel_raw

        if self.mode == "train":
            if self.perturb_accel:
                accel_rand = np.random.uniform(-1, 1, accel_calib.shape) * self.perturb_accel_range
                accel_calib += accel_rand
        self.feat = np.concatenate([accel_calib, gyro_calib, rotor_spd], axis=1)
        self.targ_vb = np.zeros((traj_target.shape[0], 3))
        for i in range(traj_target.shape[0]):
            self.targ_vb[i, :] = pose.xyzwQuatToMat(traj_target[i, 3:7]).T @ traj_target[i,7:10]
        self.traj_target = traj_target

    def get_feature(self):
        return self.feat

    def get_target(self):
        return self.targ_vb, self.traj_target

    # Auxiliary quantities, not used for training.
    def get_aux(self):
        return self.ts, self.gyro_raw, self.accel_raw

class ModelOur2Dataset(Dataset):
    """
        Dataset for our ros2 dataset.
    """
    def __init__(self, data_list, args, data_window_config, **kwargs):
        super(ModelOur2Dataset, self).__init__()

        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]
        self.g = np.array([0., 0., 9.7946])

        self.mode = kwargs.get("mode", "train")
        self.perturb_accel = args.perturb_accel
        self.perturb_accel_range = args.perturb_accel_range

        self.shuffle = False
        if self.mode == "train":
            self.shuffle = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False

        # index_map = [[seq_id, index of the last datapoint in the window], ...]
        self.index_map = []
        self.ts, self.features, self.targets, self.gt_traj = [], [], [], []
        self.raw_gyro_meas = []
        self.raw_accel_meas = []

        for i in range(len(data_list)):
            seq = ModelSequence(data_list[i], args, **kwargs)

            feat = seq.get_feature()
            targ, traj = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
            self.gt_traj.append(traj)
            N = self.features[i].shape[0]
            self.index_map += [
                [i, j]
                for j in range(
                    int(self.window_size * self.sampling_factor), 
                    N,
                    self.window_shift_size) 
                ]

            times, raw_gyro_meas, raw_accel_meas = seq.get_aux()
            self.ts.append(times)

            if self.mode == "test":
                self.raw_gyro_meas.append(raw_gyro_meas)
                self.raw_accel_meas.append(raw_accel_meas)
                
        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        idxs = frame_id - self.window_size * self.sampling_factor
        idxe = frame_id
        indices = range(idxs, idxe, self.sampling_factor)
        idxs = indices[0]
        idxe = indices[-1]

        feat = self.features[seq_id][indices]

       # target velocity
        targ = self.targets[seq_id][idxe, :]

        gt_traj = self.gt_traj[seq_id][indices]

        # auxiliary variables
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        raw_accel_meas_i = np.zeros((3,))

        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            raw_accel_meas_i = self.raw_accel_meas[seq_id][indices]
            
        return feat.astype(np.float32).T, targ.astype(np.float32), gt_traj.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, raw_accel_meas_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)

class ModelDIDODataset(Dataset):
    """
        Dataset for DIDO dataset.
    """
    def __init__(self, data_list, args, data_window_config, **kwargs):
        super(ModelDIDODataset, self).__init__()

        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]
        self.g = np.array([0., 0., 9.7946])

        self.mode = kwargs.get("mode", "train")
        self.perturb_accel = args.perturb_accel
        self.perturb_accel_range = args.perturb_accel_range

        self.shuffle = False
        if self.mode == "train":
            self.shuffle = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False

        # index_map = [[seq_id, index of the last datapoint in the window], ...]
        self.index_map = []
        self.ts, self.features, self.targets, self.gt_traj = [], [], [], []
        self.raw_gyro_meas = []
        self.raw_accel_meas = []

        for i in range(len(data_list)):
            seq = ModelSequence(data_list[i], args, **kwargs)

            feat = seq.get_feature()
            targ, traj = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
            self.gt_traj.append(traj)
            N = self.features[i].shape[0]
            self.index_map += [
                [i, j]
                for j in range(
                    int(self.window_size * self.sampling_factor), 
                    N,
                    self.window_shift_size) 
                ]

            times, raw_gyro_meas, raw_accel_meas = seq.get_aux()
            self.ts.append(times)

            if self.mode == "test":
                self.raw_gyro_meas.append(raw_gyro_meas)
                self.raw_accel_meas.append(raw_accel_meas)
                
        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        idxs = frame_id - self.window_size * self.sampling_factor
        idxe = frame_id
        indices = range(idxs, idxe, self.sampling_factor)
        idxs = indices[0]
        idxe = indices[-1]

        feat = self.features[seq_id][indices]

       # target velocity
        targ = self.targets[seq_id][idxe, :]

        gt_traj = self.gt_traj[seq_id][indices]

        # auxiliary variables
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        raw_accel_meas_i = np.zeros((3,))

        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            raw_accel_meas_i = self.raw_accel_meas[seq_id][indices]
            
        return feat.astype(np.float32).T, targ.astype(np.float32), gt_traj.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, raw_accel_meas_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)
