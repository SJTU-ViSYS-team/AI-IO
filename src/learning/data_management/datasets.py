"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/dataloader/dataset_fb.py
"""

from abc import ABC, abstractmethod
import os
import random

import h5py
import numpy as np
from torch.utils.data import Dataset

import learning.utils.pose as pose
import pypose


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
        self.perturb_orientation = args.perturb_orientation
        self.perturb_orientation_theta_range = args.perturb_orientation_theta_range
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
            # throttle = np.copy(f["throttle"])
            rotor_spd = np.copy(f["rotor_spd"])

        #[NOTE] rotate to world frame (keep for future use)
        # w_gyro_calib = np.array([pose.xyzwQuatToMat(T_wi[3:]) @ w_i for T_wi, w_i in zip(traj_target, gyro_calib)])
        # w_accel_calib = np.array([pose.xyzwQuatToMat(T_wi[3:]) @ a_i for T_wi, a_i in zip(traj_target, accel_calib)])

        self.ts = ts
        self.gyro_raw = gyro_raw
        self.accel_raw = accel_raw

        ypr = np.array([pose.fromQuatToEulerAng(targ[3:7]) for targ in traj_target]) * np.pi / 180.0
        atti = np.array([pose.xyzwQuatToMat(targ[3:7]).reshape(9)[:6] for targ in traj_target])
        if self.mode == "train":
            if self.perturb_orientation:
                theta_rand = np.random.uniform(-1, 1, ypr.shape) * np.pi * self.perturb_orientation_theta_range / 180.0
                ypr += theta_rand
                atti = np.array([pose.fromEulerAngToRotMat(ang[0], ang[1], ang[2]).reshape(9)[:6] for ang in ypr])
            if self.perturb_accel:
                accel_rand = np.random.uniform(-1, 1, accel_calib.shape) * self.perturb_accel_range
                accel_calib += accel_rand
        self.feat = np.concatenate([accel_calib, gyro_calib, rotor_spd, atti], axis=1)
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

class ModelEurocDataset(Dataset):
    """
        Dataset for Euroc dataset.
    """
    def __init__(self, data_list, args, data_window_config, **kwargs):
        super(ModelEurocDataset, self).__init__()

        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]
        self.g = np.array([0., 0., 9.8082])

        self.mode = kwargs.get("mode", "train")
        self.perturb_orientation = args.perturb_orientation
        self.perturb_orientation_theta_range = args.perturb_orientation_theta_range
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
        self.ts, self.features, self.targets = [], [], []
        self.raw_gyro_meas = []
        self.raw_accel_meas = []
        # self.thrusts = []
        for i in range(len(data_list)):
            seq = ModelSequence(data_list[i], **kwargs)
            feat = seq.get_feature()
            targ = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
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
        targ = self.targets[seq_id][idxe, 7:10]

        # auxiliary variables
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        raw_accel_meas_i = np.zeros((3,))

        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            raw_accel_meas_i = self.raw_accel_meas[seq_id][indices]
            
        if self.mode == "train":
            if self.perturb_orientation:
                theta_rand = np.random.uniform(-1, 1, feat[:, :3].shape) * np.pi * self.perturb_orientation_theta_range / 180.0
                feat[:, 0:3] += theta_rand
            if self.perturb_accel:
                accel_rand = np.random.uniform(-1, 1, feat[:, 3:6].shape) * self.perturb_accel_range
                feat[:, 3:6] += accel_rand
            
        return feat.astype(np.float32).T, targ.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, raw_accel_meas_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)
    
class ModelBlackbirdDataset(Dataset):
    """
        Dataset for Blackbird dataset.
    """
    def __init__(self, data_list, args, data_window_config, **kwargs):
        super(ModelBlackbirdDataset, self).__init__()

        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]
        self.g = np.array([0., 0., 9.81])

        self.mode = kwargs.get("mode", "train")
        self.perturb_orientation = args.perturb_orientation
        self.perturb_orientation_theta_range = args.perturb_orientation_theta_range
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
        self.ts, self.features, self.targets = [], [], []
        self.raw_gyro_meas = []
        self.raw_accel_meas = []
        # self.thrusts = []
        for i in range(len(data_list)):
            seq = ModelSequence(data_list[i], **kwargs)
            # feat = np.array([[yaw, picth, roll, ax, ay, az], ...])
            # targ = np.array([[x, y, z, qx, qy, qz, qw], ...])
            feat = seq.get_feature()
            targ = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
            N = self.features[i].shape[0]
            self.index_map += [
                [i, j]
                for j in range(
                    int(self.window_size * self.sampling_factor), # 一个window内imu的数量 
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

       # target velocity ([NOTE] only x, y)
        targ = self.targets[seq_id][idxe, 7:10]

        # auxiliary variables
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        raw_accel_meas_i = np.zeros((3,))

        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            raw_accel_meas_i = self.raw_accel_meas[seq_id][indices]
            
        if self.mode == "train":
            if self.perturb_orientation:
                theta_rand = np.random.uniform(-1, 1, feat[:, :3].shape) * np.pi * self.perturb_orientation_theta_range / 180.0
                feat[:, 0:3] += theta_rand
            if self.perturb_accel:
                accel_rand = np.random.uniform(-1, 1, feat[:, 3:6].shape) * self.perturb_accel_range
                feat[:, 3:6] += accel_rand
            
        return feat.astype(np.float32).T, targ.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, raw_accel_meas_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)

class ModelFPVDataset(Dataset):
    """
        Dataset for FPV dataset.
    """
    def __init__(self, data_list, args, data_window_config, **kwargs):
        super(ModelFPVDataset, self).__init__()

        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]
        self.g = np.array([0., 0., 9.8082])

        self.mode = kwargs.get("mode", "train")
        self.perturb_orientation = args.perturb_orientation
        self.perturb_orientation_theta_range = args.perturb_orientation_theta_range
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
        self.ts, self.features, self.targets = [], [], []
        self.raw_gyro_meas = []
        self.raw_accel_meas = []
        # self.thrusts = []
        for i in range(len(data_list)):
            seq = ModelSequence(data_list[i], **kwargs)
            # feat = np.array([[yaw, picth, roll, ax, ay, az], ...])
            # targ = np.array([[x, y, z, qx, qy, qz, qw], ...])
            feat = seq.get_feature()
            targ = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
            N = self.features[i].shape[0]
            self.index_map += [
                [i, j]
                for j in range(
                    int(self.window_size * self.sampling_factor), # 一个window内imu的数量 
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

       # target velocity ([NOTE] only x, y)
        targ = self.targets[seq_id][idxe, 7:10]

        # auxiliary variables
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        raw_accel_meas_i = np.zeros((3,))

        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            raw_accel_meas_i = self.raw_accel_meas[seq_id][indices]
            
        if self.mode == "train":
            if self.perturb_orientation:
                theta_rand = np.random.uniform(-1, 1, feat[:, :3].shape) * np.pi * self.perturb_orientation_theta_range / 180.0
                feat[:, 0:3] += theta_rand
            if self.perturb_accel:
                accel_rand = np.random.uniform(-1, 1, feat[:, 3:6].shape) * self.perturb_accel_range
                feat[:, 3:6] += accel_rand
            
        return feat.astype(np.float32).T, targ.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, raw_accel_meas_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)
    
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
        self.perturb_orientation = args.perturb_orientation
        self.perturb_orientation_theta_range = args.perturb_orientation_theta_range
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
        # self.thrusts = []
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
            
        # if self.mode == "train":
        #     if self.perturb_orientation:
        #         theta_rand = np.random.uniform(-1, 1, feat[:, :3].shape) * np.pi * self.perturb_orientation_theta_range / 180.0
        #         ypr += theta_rand
        #         feat[:, 10:] = np.array([pose.fromEulerAngToRotMat(ang[0], ang[1], ang[2]).reshape(9)[:6] for ang in ypr])
        #     if self.perturb_accel:
        #         accel_rand = np.random.uniform(-1, 1, feat[:, 3:6].shape) * self.perturb_accel_range
        #         feat[:, :3] += accel_rand
            
        return feat.astype(np.float32).T, targ.astype(np.float32), gt_traj.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, raw_accel_meas_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)

class ModelSimulationDataset(Dataset):
    """
        Dataset for Isaac Simulation dataset
    """
    def __init__(self) -> None:
        super().__init__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self) -> int:
        pass

class ModelDIDODataset(Dataset):
    """
        Dataset for Dido dataset.
    """
    def __init__(self, data_list, args, data_window_config, **kwargs):
        super(ModelDIDODataset, self).__init__()

        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]
        self.g = np.array([0., 0., 9.7946])

        self.mode = kwargs.get("mode", "train")
        self.perturb_orientation = args.perturb_orientation
        self.perturb_orientation_theta_range = args.perturb_orientation_theta_range
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
        # self.thrusts = []
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
            
        # if self.mode == "train":
        #     if self.perturb_orientation:
        #         theta_rand = np.random.uniform(-1, 1, feat[:, :3].shape) * np.pi * self.perturb_orientation_theta_range / 180.0
        #         ypr += theta_rand
        #         feat[:, 10:] = np.array([pose.fromEulerAngToRotMat(ang[0], ang[1], ang[2]).reshape(9)[:6] for ang in ypr])
        #     if self.perturb_accel:
        #         accel_rand = np.random.uniform(-1, 1, feat[:, 3:6].shape) * self.perturb_accel_range
        #         feat[:, :3] += accel_rand
            
        return feat.astype(np.float32).T, targ.astype(np.float32), gt_traj.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, raw_accel_meas_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)

class ModelOursDataset(Dataset):
    """
        Dataset for ours dataset.
    """
    def __init__(self, data_list, args, data_window_config, **kwargs):
        super(ModelOursDataset, self).__init__()

        self.sampling_factor = data_window_config["sampling_factor"]
        self.window_size = int(data_window_config["window_size"])
        self.window_shift_size = data_window_config["window_shift_size"]
        self.g = np.array([0., 0., 9.7964])

        self.mode = kwargs.get("mode", "train")
        self.perturb_orientation = args.perturb_orientation
        self.perturb_orientation_theta_range = args.perturb_orientation_theta_range
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
        self.ts, self.features, self.targets = [], [], []
        self.raw_gyro_meas = []
        self.raw_accel_meas = []
        # self.thrusts = []
        for i in range(len(data_list)):
            seq = ModelSequence(data_list[i], **kwargs)
            # feat = np.array([[yaw, picth, roll, ax, ay, az], ...])
            # targ = np.array([[x, y, z, qx, qy, qz, qw], ...])
            feat = seq.get_feature()
            targ = seq.get_target()
            self.features.append(feat)
            self.targets.append(targ)
            N = self.features[i].shape[0]
            self.index_map += [
                [i, j]
                for j in range(
                    int(self.window_size * self.sampling_factor), # 一个window内imu的数量 
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

       # target velocity ([NOTE] only x, y)
        targ = self.targets[seq_id][idxe, 7:10]

        # auxiliary variables
        feat_ts = self.ts[seq_id][indices]

        raw_gyro_meas_i = np.zeros((3,))
        raw_accel_meas_i = np.zeros((3,))

        if self.mode == "test":
            raw_gyro_meas_i = self.raw_gyro_meas[seq_id][indices]
            raw_accel_meas_i = self.raw_accel_meas[seq_id][indices]
            
        if self.mode == "train":
            if self.perturb_orientation:
                theta_rand = np.random.uniform(-1, 1, feat[:, :3].shape) * np.pi * self.perturb_orientation_theta_range / 180.0
                feat[:, 0:3] += theta_rand
            if self.perturb_accel:
                accel_rand = np.random.uniform(-1, 1, feat[:, 3:6].shape) * self.perturb_accel_range
                feat[:, 3:6] += accel_rand
            
        return feat.astype(np.float32).T, targ.astype(np.float32), \
            feat_ts, raw_gyro_meas_i.astype(np.float32).T, raw_accel_meas_i.astype(np.float32).T

    def __len__(self):
        return len(self.index_map)