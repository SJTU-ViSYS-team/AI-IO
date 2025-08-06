import os
import argparse
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

from mcap.reader import make_reader
from mcap_ros2.decoder import Decoder
import utils
from pyhocon import ConfigFactory

'''
    python src/learning/data_management/prepare_datasets/our2.py --config config/our2.conf
'''
# NOTE: 
# the provided ground truth is the drone body in the NWU vicon frame
# rotate to have z upwards, to NWU
# R_w_nwu = np.array([
#     [1., 0., 0.],
#     [0., 1., 0.],
#     [0., 0., 1.]])
# t_w_nwu = np.array([0., 0., 0.])

# # rotate from imu to body frame, f-l-u
# R_b_i = np.array([
#     [1., 0., 0.],
#     [0., 1., 0.],
#     [0., 0., 1.]])
# t_b_i = np.array([0., 0., 0.])

# # w1 to w2: UEN to NWU
# R_w2_w1 = np.array([
#     [1., 0., 0.],
#     [0., 1., 0.],
#     [0., 0., 1.]
# ])
# t_w2_w1 = np.array([0., 0., 0.])

dt = 0.01

def process_sequence(dataset_dir, seq_name, save_txt, split_ratios=(0.7, 0.15, 0.15)):
    # base_seq_name = os.path.dirname(os.path.dirname(seq_name))
    data_dir = os.path.join(dataset_dir, seq_name)
    assert os.path.isdir(data_dir), '%s' % data_dir

    mcap_files = [f for f in os.listdir(data_dir) if f.endswith('.mcap')]
    if len(mcap_files) == 0:
        raise FileNotFoundError("No .mcap file found in the directory.")
    elif len(mcap_files) > 1:
        raise RuntimeError(f"Multiple .mcap files found: {mcap_files}, please specify one.")
    else:
        mcap_fn = os.path.join(data_dir, mcap_files[0])

    # Read data
    raw_imu = []  # [ts wx wy wz ax ay az]
    pose_gt = []
    throt = []
    esc = []

    imu_topic = '/mavros/imu/data'
    pose_topic = '/odom/global'
    throt_topic = '/mavros/vfr_hud'
    esc_topic = '/mavros/esc'

    print('Reading data from %s' % mcap_fn)
    decoder = Decoder()
    with open(mcap_fn, "rb") as f:
        reader = make_reader(f)
        for schema, channel, msg in reader.iter_messages():
            if channel.topic == imu_topic and channel.message_encoding == "cdr":
                try:
                    ros_msg = decoder.decode(schema, msg)
                except Exception as e:
                    print(f"[Decode error] {e}")
                    continue
                imu_i = np.array([
                    ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec*1e-9,
                    ros_msg.angular_velocity.x, ros_msg.angular_velocity.y, ros_msg.angular_velocity.z,
                    ros_msg.linear_acceleration.x, ros_msg.linear_acceleration.y, ros_msg.linear_acceleration.z])
                raw_imu.append(imu_i)
                
            elif channel.topic == pose_topic and channel.message_encoding == "cdr":
                try:
                    ros_msg = decoder.decode(schema, msg)
                except Exception as e:
                    print(f"[Decode error] {e}")
                    continue
                pose_i = np.array([
                    ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec*1e-9,
                    ros_msg.pose.pose.position.x, ros_msg.pose.pose.position.y, ros_msg.pose.pose.position.z,
                    ros_msg.pose.pose.orientation.x, ros_msg.pose.pose.orientation.y, ros_msg.pose.pose.orientation.z, ros_msg.pose.pose.orientation.w,
                    ros_msg.twist.twist.linear.x, ros_msg.twist.twist.linear.y, ros_msg.twist.twist.linear.z
                ])
                pose_gt.append(pose_i)

            elif channel.topic == throt_topic and channel.message_encoding == "cdr":
                try:
                    ros_msg = decoder.decode(schema, msg)
                except Exception as e:
                    print(f"[Decode error] {e}")
                    continue
                throt_i = np.array([
                    ros_msg.header.stamp.sec + ros_msg.header.stamp.nanosec*1e-9,
                    ros_msg.throttle
                ])
                throt.append(throt_i)

            elif channel.topic == esc_topic and channel.message_encoding == "cdr":
                try:
                    ros_msg = decoder.decode(schema, msg)
                except Exception as e:
                    print(f"[Decode error] {e}")
                    continue
                esc_i = np.array([
                    ros_msg.esc_status[0].header.stamp.sec + ros_msg.esc_status[0].header.stamp.nanosec*1e-9,
                    ros_msg.esc_status[0].rpm, ros_msg.esc_status[1].rpm, ros_msg.esc_status[2].rpm, ros_msg.esc_status[3].rpm
                ])
                esc.append(esc_i)

    raw_imu = np.asarray(raw_imu)
    pose_gt = np.asarray(pose_gt)
    throt = np.asarray(throt)
    esc = np.asarray(esc)

    gt_traj_tmp = pose_gt

    t_curr = raw_imu[0, 0]
    new_times_imu = [t_curr]
    while t_curr < raw_imu[-1, 0] - dt - 0.0001:
        t_curr = t_curr + dt
        new_times_imu.append(t_curr)
    new_times_imu = np.asarray(new_times_imu) # fixed frequency time sequence
    gyro_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 1:4], axis=0)(new_times_imu)
    accel_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 4:7], axis=0)(new_times_imu)
    raw_imu = np.concatenate((new_times_imu.reshape((-1, 1)), gyro_tmp, accel_tmp), axis=1)

    # We down sample to IMU rate
    times_imu = raw_imu[:, 0]
    # get initial and final times for interpolations
    idx_s = 0
    for ts in times_imu:
        if ts > gt_traj_tmp[0, 0] and ts > throt[0, 0] and ts > esc[0, 0]:
            break
        else:
            idx_s = idx_s + 1
    assert idx_s < len(times_imu)

    idx_e = len(times_imu) - 1
    for ts in reversed(times_imu):
        if ts < gt_traj_tmp[-1, 0] and ts < throt[-1, 0] and ts < esc[-1, 0]:
            break
        else:
            idx_e = idx_e - 1
    assert idx_e > 0

    times_imu = times_imu[idx_s:idx_e + 1]
    raw_imu = raw_imu[idx_s:idx_e + 1]
    start_time, end_time = times_imu[0], times_imu[-1]

    # interpolate ground-truth samples at imu times
    throt_data = interp1d(throt[:, 0], throt[:, 1],axis=0)(times_imu)
    esc_data = interp1d(esc[:, 0], esc[:, 1:],axis=0)(times_imu)
    groundtruth_pos_data = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 1:4], axis=0)(times_imu)
    groundtruth_rot_data = Slerp(gt_traj_tmp[:, 0], Rotation.from_quat(gt_traj_tmp[:, 4:8]))(times_imu)
    groundtruth_vel_data = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 8:11], axis=0)(times_imu)

    throttle_data = np.concatenate((
        times_imu.reshape((-1, 1)),
        throt_data.reshape((-1, 1))
    ),axis=1)
    rotor_spd_data = np.concatenate((
        times_imu.reshape((-1, 1)),
        esc_data
    ),axis=1)
    gt_traj = np.concatenate((times_imu.reshape((-1, 1)),
                                groundtruth_pos_data,
                                groundtruth_rot_data.as_quat(),
                                groundtruth_vel_data), axis=1)

    ts = raw_imu[:, 0]

    # Calibrate
    imu_calibrator = utils.getImuCalib("Blackbird")
    b_g = imu_calibrator["gyro_bias"]
    b_a = imu_calibrator["accel_bias"]
    w_calib = raw_imu[:, 1:4].T - b_g[:, None]
    a_calib = raw_imu[:, 4:].T - b_a[:, None]
    calib_imu = np.concatenate((raw_imu[:, 0].reshape((-1, 1)), w_calib.T, a_calib.T), axis=1)

    total_len = len(ts)
    raw_splits = [int(r * total_len) for r in split_ratios]
    raw_splits[-1] = total_len - sum(raw_splits[:-1]) 
    split_points = np.cumsum(raw_splits)
    split_names = ['train', 'val', 'test']

    prev_idx = 0
    for name, end_idx in zip(split_names, split_points):
        if end_idx - prev_idx == 0:
            prev_idx = end_idx
            continue

        ts_sub = ts[prev_idx:end_idx]
        raw_imu_sub = raw_imu[prev_idx:end_idx]
        calib_imu_sub = calib_imu[prev_idx:end_idx]
        gt_traj_sub = gt_traj[prev_idx:end_idx]
        throttle_sub = throttle_data[prev_idx:end_idx]
        rotor_spd_sub = rotor_spd_data[prev_idx:end_idx]

        out_dir = os.path.join(dataset_dir, seq_name, 'processed_data', name)
        os.makedirs(out_dir, exist_ok=True)
        out_fn = os.path.join(out_dir, "data.hdf5")
        with h5py.File(out_fn, "w") as f:
            f.create_dataset("ts", data=ts_sub)
            f.create_dataset("gyro_raw", data=raw_imu_sub[:, 1:4])
            f.create_dataset("accel_raw", data=raw_imu_sub[:, 4:])
            f.create_dataset("gyro_calib", data=calib_imu_sub[:, 1:4])
            f.create_dataset("accel_calib", data=calib_imu_sub[:, 4:])
            f.create_dataset("traj_target", data=gt_traj_sub[:, 1:11])
            f.create_dataset("gyro_bias", data=b_g)
            f.create_dataset("accel_bias", data=b_a)
            f.create_dataset("throttle", data=throttle_sub[:, 1:])
            f.create_dataset("rotor_spd", data=rotor_spd_sub[:, 1:])

        print(f"[{name}] data is saved to {out_fn}")

        if save_txt:
            txt_path = os.path.join(out_dir, "stamped_groundtruth_imu.txt")
            np.savetxt(
                txt_path, gt_traj_sub, fmt="%.6f",
                header="ts x y z qx qy qz qw vx vy vz"
            )
            print(f"[{name}] groundtruth is saved to {txt_path}")
        prev_idx = end_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare dataset to HDF5 data")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--save_txt", action="store_true", help="if save txt file", default=True)
    args = parser.parse_args()

    config = ConfigFactory.parse_file(args.config)

    split_config = config.get('data_pre', {})

    for dataset in split_config["data_list"]:
        dataset_dir = dataset["data_root"]
        proportion = tuple(map(float, dataset["data_proportion"]))
        for seq_name in dataset["data_drive"]:
            print(f"prepare data: {seq_name}")
            process_sequence(
                dataset_dir=dataset_dir,
                seq_name=seq_name,
                save_txt=args.save_txt,
                split_ratios=proportion
            )

