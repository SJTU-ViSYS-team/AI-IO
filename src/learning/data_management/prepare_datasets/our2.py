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
# NOTE: 坐标系一致，不需要转换
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

# dt = 0.01

def process_sequence(dataset_dir, seq_name, mode, save_txt):
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
    # rosbag_fn = os.path.join(data_dir, 'rosbag.bag')

    # Read data
    raw_imu = []  # [ts wx wy wz ax ay az]
    pose_gt = []
    dt = 0.01

    imu_topic = '/mavros/imu/data'
    pose_topic = '/odom/global'

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

    raw_imu = np.asarray(raw_imu)
    pose_gt = np.asarray(pose_gt)

    gt_traj_tmp = pose_gt

    # # include velocities
    # gt_times = pose_gt[:, 0]
    # gt_pos = pose_gt[:, 1:4]

    # # compute velocity
    # v_start = ((gt_pos[1] - gt_pos[0]) / (gt_times[1] - gt_times[0])).reshape((1, 3))
    # gt_vel_raw = (gt_pos[1:] - gt_pos[:-1]) / (gt_times[1:] - gt_times[:-1])[:, None]
    # gt_vel_raw = np.concatenate((v_start, gt_vel_raw), axis=0)
    # # filter
    # gt_vel_x = np.convolve(gt_vel_raw[:, 0], np.ones(5) / 5, mode='same')
    # gt_vel_x = gt_vel_x.reshape((-1, 1))
    # gt_vel_y = np.convolve(gt_vel_raw[:, 1], np.ones(5) / 5, mode='same')
    # gt_vel_y = gt_vel_y.reshape((-1, 1))
    # gt_vel_z = np.convolve(gt_vel_raw[:, 2], np.ones(5) / 5, mode='same')
    # gt_vel_z = gt_vel_z.reshape((-1, 1))
    # gt_vel = np.concatenate((gt_vel_x, gt_vel_y, gt_vel_z), axis=1)

    # gt_traj_tmp = np.concatenate((pose_gt, gt_vel), axis=1)  # [ts x y z qx qy qz qw vx vy vz]

    # In FPV dataset, the sensors measurements are at:
    # 500 Hz IMU meas.
    # resample imu at exactly 100 Hz
    # dt = 0.01
    t_curr = raw_imu[0, 0]
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
    start_time, end_time = times_imu[0], times_imu[-1]
    times_config = {
        mode: [start_time, end_time]
    }

    # interpolate ground-truth samples at imu times
    groundtruth_pos_data = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 1:4], axis=0)(times_imu)
    groundtruth_rot_data = Slerp(gt_traj_tmp[:, 0], Rotation.from_quat(gt_traj_tmp[:, 4:8]))(times_imu)
    groundtruth_vel_data = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 8:11], axis=0)(times_imu)
    groundtruth_rot_data_inv = groundtruth_rot_data.inv()
    # prepare vel in b frame
    # groundtruth_vel_data_b = groundtruth_rot_data_inv.apply(groundtruth_vel_data)

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

    # sample relevant times
    ts0, ts1 = times_config[mode]
    idx0_candidates = np.where(ts >= ts0)[0]
    idx1_candidates = np.where(ts >= ts1)[0]

    if len(idx0_candidates) == 0 or len(idx1_candidates) == 0:
        raise ValueError(f"No valid index found for mode={mode}: ts0={ts0}, ts1={ts1}, ts range=({ts[0]}, {ts[-1]})")

    idx0 = idx0_candidates[0]
    idx1 = idx1_candidates[0]

    ts_mode = ts[idx0:idx1]
    raw_imu_mode = raw_imu[idx0:idx1]
    calib_imu_mode = calib_imu[idx0:idx1]
    gt_traj_mode = gt_traj[idx0:idx1]

    # Not supported on this branch
    # traj_target_oris_from_imu_list = []
    # traj_target_oris_from_imu_list.append(gt_traj[0])
    # traj_target_oris_from_imu = np.asarray(traj_target_oris_from_imu_list)

    # Save
    out_dir = os.path.join(data_dir, 'processed_data', mode)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fn = os.path.join(out_dir, "data.hdf5")
    with h5py.File(out_fn, "w") as f:
        ts = f.create_dataset("ts", data=ts_mode)
        gyro_raw = f.create_dataset("gyro_raw", data=raw_imu_mode[:, 1:4])
        accel_raw = f.create_dataset("accel_raw", data=raw_imu_mode[:, 4:])
        gyro_calib = f.create_dataset("gyro_calib", data=calib_imu_mode[:, 1:4])
        accel_calib = f.create_dataset("accel_calib", data=calib_imu_mode[:, 4:])
        traj_target = f.create_dataset("traj_target", data=gt_traj_mode[:, 1:11])
        # traj_target_oris_from_imu_target = \
        #     f.create_dataset("traj_target_oris_from_imu", data=traj_target_oris_from_imu[:, 1:])
        gyro_bias = f.create_dataset("gyro_bias", data=b_g)
        accel_bias = f.create_dataset("accel_bias", data=b_a)

    print(f"数据保存到 {out_fn}")
    # 如果需要保存为文本文件
    if save_txt:
        txt_path = os.path.join(out_dir, "stamped_groundtruth_imu.txt")
        np.savetxt(
            txt_path, gt_traj_mode[:, :8], fmt="%.6f",
            header="ts x y z qx qy qz qw"
        )
        print(f"数据保存到 {txt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理数据集并生成 HDF5 数据")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径（YAML）")
    parser.add_argument("--save_txt", action="store_true", help="是否保存为文本文件", default=True)
    args = parser.parse_args()

    config = ConfigFactory.parse_file(args.config)

    for split in ['train', 'val', 'test']:
        split_config = config.get(split, {})
        if not split_config:
            continue

        mode = split_config.get("mode", split)
        for dataset in split_config["data_list"]:
            dataset_dir = dataset["data_root"]
            for seq_name in dataset["data_drive"]:
                print(f"处理 {mode.upper()} 数据集: {seq_name}")
                process_sequence(
                    dataset_dir=dataset_dir,
                    seq_name=seq_name,
                    mode=mode,
                    save_txt=args.save_txt
                )

