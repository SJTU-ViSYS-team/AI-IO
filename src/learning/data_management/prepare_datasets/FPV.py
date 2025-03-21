"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Prepare Blackbird dataset for training, validation, and testing.
"""

import argparse
import os

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
import utils as utils
import rosbag

# NOTE: 坐标系一致，不需要转换
# the provided ground truth is the drone body in the NWU vicon frame
# rotate to have z upwards, to NWU
R_w_nwu = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]])
t_w_nwu = np.array([0., 0., 0.])

# rotate from imu to body frame, f-l-u
R_b_i = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]])
t_b_i = np.array([0., 0., 0.])

# initial and final times
train_times = {}
train_times['indoor/forward/seq_3'] = [4908.0, 4938.0]
train_times['indoor/forward/seq_5'] = [1540821126.0, 1540821145.0] # all for train
train_times['indoor/forward/seq_6'] = [1540821388.0, 1540821398.0]
train_times['indoor/forward/seq_7'] = [1540821845.0, 1540821890.0]
train_times['indoor/forward/seq_9'] = [1540822845.0, 1540822873.0] # all for train
train_times['indoor/forward/seq_10'] = [1540823082.0, 1540823092.0]

val_times = {}
val_times['indoor/forward/seq_3'] = [4938.0, 4948.0]
val_times['indoor/forward/seq_5'] = [1540821126.0, 1540821145.0]
val_times['indoor/forward/seq_6'] = [1540821398.0, 1540821408.0]
val_times['indoor/forward/seq_7'] = [1540821890.0, 1540821900.0]
val_times['indoor/forward/seq_9'] = [1540822845.0, 1540822873.0]
val_times['indoor/forward/seq_10'] = [1540823092.0, 1540823102.0]

test_times = {}
test_times['indoor/forward/seq_3'] = [4948.0, 4958.0]
test_times['indoor/forward/seq_5'] = [1540821126.0, 1540821145.0]
test_times['indoor/forward/seq_6'] = [1540821408.0, 1540821417.0]
test_times['indoor/forward/seq_7'] = [1540821900.0, 1540821911.0]
test_times['indoor/forward/seq_9'] = [1540822845.0, 1540822873.0]
test_times['indoor/forward/seq_10'] = [1540823102.0, 1540823112.0]


def prepare_dataset(args):
    dataset_dir = args.dataset_dir

    # read seq names
    seq_names = []
    seq_names.append(utils.get_datalist(os.path.join(dataset_dir, args.data_list)))
    seq_names = [item for sublist in seq_names for item in sublist]

    for idx, seq_name in enumerate(seq_names):
        # base_seq_name = os.path.dirname(os.path.dirname(seq_name))
        data_dir = os.path.join(dataset_dir, seq_name)
        assert os.path.isdir(data_dir), '%s' % data_dir
        rosbag_fn = os.path.join(data_dir, 'rosbag.bag')

        # Read data
        raw_imu = []  # [ts wx wy wz ax ay az]
        pose_gt = []
        dt = 0.002

        imu_topic = '/snappy_imu'
        pose_topic = '/groundtruth/pose'

        print('Reading data from %s' % rosbag_fn)
        with rosbag.Bag(rosbag_fn, 'r') as bag:
            for (topic, msg, ts) in bag.read_messages():
                if topic == imu_topic:
                    imu_i = np.array([
                        msg.header.stamp.to_sec(),
                        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
                    raw_imu.append(imu_i)

                elif topic == pose_topic:
                    pose_i = np.array([
                        msg.header.stamp.to_sec(),
                        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
                    ])
                    pose_gt.append(pose_i)
        # TODO: 坐标系变换

        raw_imu = np.asarray(raw_imu)
        pose_gt = np.asarray(pose_gt)

        # include velocities
        gt_times = pose_gt[:, 0]
        gt_pos = pose_gt[:, 1:4]

        # compute velocity
        v_start = ((gt_pos[1] - gt_pos[0]) / (gt_times[1] - gt_times[0])).reshape((1, 3))
        gt_vel_raw = (gt_pos[1:] - gt_pos[:-1]) / (gt_times[1:] - gt_times[:-1])[:, None]
        gt_vel_raw = np.concatenate((v_start, gt_vel_raw), axis=0)
        # filter
        gt_vel_x = np.convolve(gt_vel_raw[:, 0], np.ones(5) / 5, mode='same')
        gt_vel_x = gt_vel_x.reshape((-1, 1))
        gt_vel_y = np.convolve(gt_vel_raw[:, 1], np.ones(5) / 5, mode='same')
        gt_vel_y = gt_vel_y.reshape((-1, 1))
        gt_vel_z = np.convolve(gt_vel_raw[:, 2], np.ones(5) / 5, mode='same')
        gt_vel_z = gt_vel_z.reshape((-1, 1))
        gt_vel = np.concatenate((gt_vel_x, gt_vel_y, gt_vel_z), axis=1)

        gt_traj_tmp = np.concatenate((pose_gt, gt_vel), axis=1)  # [ts x y z qx qy qz qw vx vy vz]

        # In FPV dataset, the sensors measurements are at:
        # 500 Hz IMU meas.
        # resample imu at exactly 100 Hz
        t_curr = raw_imu[0, 0]
        # dt = 0.01
        new_times_imu = [t_curr]
        while t_curr < raw_imu[-1, 0] - dt - 0.0001:
            t_curr = t_curr + dt
            new_times_imu.append(t_curr)
        new_times_imu = np.asarray(new_times_imu) # 严格的500Hz时间序列
        gyro_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 1:4], axis=0)(new_times_imu)
        accel_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 4:7], axis=0)(new_times_imu)
        raw_imu = np.concatenate((new_times_imu.reshape((-1, 1)), gyro_tmp, accel_tmp), axis=1)

        # We down sample to IMU rate
        times_imu = raw_imu[:, 0]
        # get initial and final times for interpolations
        idx_s = 0
        for ts in times_imu:
            if ts > gt_traj_tmp[0, 0]:
                break
            else:
                idx_s = idx_s + 1
        assert idx_s < len(times_imu)

        idx_e = len(times_imu) - 1
        for ts in reversed(times_imu):
            if ts < gt_traj_tmp[-1, 0]:
                break
            else:
                idx_e = idx_e - 1
        assert idx_e > 0

        times_imu = times_imu[idx_s:idx_e + 1]
        raw_imu = raw_imu[idx_s:idx_e + 1]

        # interpolate ground-truth samples at imu times
        groundtruth_pos_data = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 1:4], axis=0)(times_imu)
        groundtruth_rot_data = Slerp(gt_traj_tmp[:, 0], Rotation.from_quat(gt_traj_tmp[:, 4:8]))(times_imu)
        groundtruth_vel_data = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 8:11], axis=0)(times_imu)
        groundtruth_rot_data_inv = groundtruth_rot_data.inv()
        # prepare vel in b frame
        groundtruth_vel_data_b = groundtruth_rot_data_inv.apply(groundtruth_vel_data)

        gt_traj = np.concatenate((times_imu.reshape((-1, 1)),
                                  groundtruth_pos_data,
                                  groundtruth_rot_data.as_quat(),
                                  groundtruth_vel_data_b), axis=1)

        ts = raw_imu[:, 0]

        # Calibrate
        imu_calibrator = utils.getImuCalib("Blackbird")
        b_g = imu_calibrator["gyro_bias"]
        b_a = imu_calibrator["accel_bias"]
        w_calib = raw_imu[:, 1:4].T - b_g[:, None]
        a_calib = raw_imu[:, 4:].T - b_a[:, None]
        calib_imu = np.concatenate((raw_imu[:, 0].reshape((-1, 1)), w_calib.T, a_calib.T), axis=1)

        # sample relevant times
        ts0_train, ts1_train = train_times[seq_name]
        idx0_train = np.where(ts > ts0_train)[0][0]
        idx1_train = np.where(ts > ts1_train)[0][0]

        ts0_val, ts1_val = val_times[seq_name]
        idx0_val = np.where(ts > ts0_val)[0][0]
        idx1_val = np.where(ts > ts1_val)[0][0]

        ts0_test, ts1_test = test_times[seq_name]
        idx0_test = np.where(ts > ts0_test)[0][0]
        idx1_test = np.where(ts > ts1_test)[0][0]

        ts_train = ts[idx0_train:idx1_train]
        raw_imu_train = raw_imu[idx0_train:idx1_train]
        calib_imu_train = calib_imu[idx0_train:idx1_train]
        gt_traj_train = gt_traj[idx0_train:idx1_train]

        ts_val = ts[idx0_val:idx1_val]
        raw_imu_val = raw_imu[idx0_val:idx1_val]
        calib_imu_val = calib_imu[idx0_val:idx1_val]
        gt_traj_val = gt_traj[idx0_val:idx1_val]

        ts_test = ts[idx0_test:idx1_test]
        raw_imu_test = raw_imu[idx0_test:idx1_test]
        calib_imu_test = calib_imu[idx0_test:idx1_test]
        gt_traj_test = gt_traj[idx0_test:idx1_test]

        # Not supported on this branch
        # traj_target_oris_from_imu_list = []
        # traj_target_oris_from_imu_list.append(gt_traj[0])
        # traj_target_oris_from_imu = np.asarray(traj_target_oris_from_imu_list)

        # Save
        # train
        out_dir = os.path.join(data_dir, "train")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_fn = os.path.join(out_dir, "data.hdf5")
        with h5py.File(out_fn, "w") as f:
            ts = f.create_dataset("ts", data=ts_train)
            gyro_raw = f.create_dataset("gyro_raw", data=raw_imu_train[:, 1:4])
            accel_raw = f.create_dataset("accel_raw", data=raw_imu_train[:, 4:])
            gyro_calib = f.create_dataset("gyro_calib", data=calib_imu_train[:, 1:4])
            accel_calib = f.create_dataset("accel_calib", data=calib_imu_train[:, 4:])
            traj_target = f.create_dataset("traj_target", data=gt_traj_train[:, 1:11])
            # traj_target_oris_from_imu_target = \
            #     f.create_dataset("traj_target_oris_from_imu", data=traj_target_oris_from_imu[:, 1:])
            gyro_bias = f.create_dataset("gyro_bias", data=b_g)
            accel_bias = f.create_dataset("accel_bias", data=b_a)

        if args.save_txt:
            np.savetxt(os.path.join(out_dir, "imu_raw.txt"),
                       raw_imu_train, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "imu_calib.txt"),
                       calib_imu_train, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "stamped_groundtruth_imu.txt"),
                       gt_traj_train, fmt='%.12f', header='ts x y z qx qy qz qw')

        print("File data.hdf5 written to " + out_fn)

        # val
        out_dir = os.path.join(data_dir, "val")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_fn = os.path.join(out_dir, "data.hdf5")
        with h5py.File(out_fn, "w") as f:
            ts = f.create_dataset("ts", data=ts_val)
            gyro_raw = f.create_dataset("gyro_raw", data=raw_imu_val[:, 1:4])
            accel_raw = f.create_dataset("accel_raw", data=raw_imu_val[:, 4:])
            gyro_calib = f.create_dataset("gyro_calib", data=calib_imu_val[:, 1:4])
            accel_calib = f.create_dataset("accel_calib", data=calib_imu_val[:, 4:])
            traj_target = f.create_dataset("traj_target", data=gt_traj_val[:, 1:11])
            # traj_target_oris_from_imu_target = \
            #     f.create_dataset("traj_target_oris_from_imu", data=traj_target_oris_from_imu[:, 1:])
            gyro_bias = f.create_dataset("gyro_bias", data=b_g)
            accel_bias = f.create_dataset("accel_bias", data=b_a)

        if args.save_txt:
            np.savetxt(os.path.join(out_dir, "imu_raw.txt"),
                       raw_imu_val, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "imu_calib.txt"),
                       calib_imu_val, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "stamped_groundtruth_imu.txt"),
                       gt_traj_val, fmt='%.12f', header='ts x y z qx qy qz qw')

        print("File data.hdf5 written to " + out_fn)

        # test
        out_dir = os.path.join(data_dir, "test")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_fn = os.path.join(out_dir, "data.hdf5")
        with h5py.File(out_fn, "w") as f:
            ts = f.create_dataset("ts", data=ts_test)
            gyro_raw = f.create_dataset("gyro_raw", data=raw_imu_test[:, 1:4])
            accel_raw = f.create_dataset("accel_raw", data=raw_imu_test[:, 4:])
            gyro_calib = f.create_dataset("gyro_calib", data=calib_imu_test[:, 1:4])
            accel_calib = f.create_dataset("accel_calib", data=calib_imu_test[:, 4:])
            traj_target = f.create_dataset("traj_target", data=gt_traj_test[:, 1:11])
            # traj_target_oris_from_imu_target = \
            #     f.create_dataset("traj_target_oris_from_imu", data=traj_target_oris_from_imu[:, 1:])
            gyro_bias = f.create_dataset("gyro_bias", data=b_g)
            accel_bias = f.create_dataset("accel_bias", data=b_a)

        if args.save_txt:
            np.savetxt(os.path.join(out_dir, "imu_raw.txt"),
                       raw_imu_test, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "imu_calib.txt"),
                       calib_imu_test, fmt='%.12f', header='ts wx wy wz ax ay az')
            np.savetxt(os.path.join(out_dir, "stamped_groundtruth_imu.txt"),
                       gt_traj_test, fmt='%.12f', header='ts x y z qx qy qz qw')

        print("File data.hdf5 written to " + out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--data_list", type=str)
    parser.add_argument("--save_txt", action="store_true", default=True)
    args = parser.parse_args()

    prepare_dataset(args)

