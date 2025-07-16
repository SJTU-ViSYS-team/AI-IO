import os
import argparse
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

import utils
from pyhocon import ConfigFactory

'''
    python src/learning/data_management/prepare_datasets/euroc.py --config config/Euroc.conf
'''
# # the provided ground truth is the drone body in the NED vicon frame
# # rotate to have z upwards, NED to NWU
# R_w_ned = np.array([
#     [1., 0., 0.],
#     [0., -1., 0.],
#     [0., 0., -1.]])
# t_w_ned = np.array([0., 0., 0.])

# # rotate from body to imu frame
# R_b_i = np.array([
#     [0., -1., 0.],
#     [1., 0., 0.],
#     [0., 0., 1.]])
# t_b_i = np.array([0., 0., 0.])

# body coordinate is the same as imu
# R_b_i = np.array([
#     [0., 0., 1.],
#     [0., -1., 0.],
#     [1., 0., 0.]
# ])
# t_b_i = np.array([0., 0., 0.])

# w1 to w2: UEN to NWU
R_w2_w1 = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
])
t_w2_w1 = np.array([0., 0., 0.])

dt = 0.005

def process_sequence(dataset_dir, seq_name, mode, save_txt):
    seq_dir = os.path.join(dataset_dir, seq_name)
    imu_csv = os.path.join(seq_dir, "imu0/data.csv")
    pose_csv = os.path.join(seq_dir, "state_groundtruth_estimate0/data.csv")

    assert os.path.isfile(imu_csv), f"IMU 数据文件缺失：{imu_csv}"
    assert os.path.isfile(pose_csv), f"位姿数据文件缺失：{pose_csv}"

    # 读取 imu 时间范围
    imu_data = pd.read_csv(imu_csv)
    imu_times = imu_data['#timestamp [ns]'].values / 1e9
    start_time, end_time = imu_times[0], imu_times[-1]
    times_config = {
        mode: [start_time, end_time]
    }

    output_dir = os.path.join(seq_dir, "processed_data")

    # 调用数据处理函数
    process_and_save_to_hdf5(
        imu_csv=imu_csv,
        pose_csv=pose_csv,
        output_dir=output_dir,
        sequence_name=seq_name,
        times_config=times_config,
        save_txt=save_txt
    )

def process_and_save_to_hdf5(imu_csv, pose_csv, output_dir, sequence_name, times_config, save_txt=False):
    """
    处理 IMU 和位姿数据，生成训练/验证/测试集并存储为 HDF5 文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imu_data = pd.read_csv(imu_csv)
    pose_data = pd.read_csv(pose_csv)
    pose_data = pose_data.apply(process_pose_data_row, axis=1)
    # print(imu_data.columns)
    imu_times = imu_data['#timestamp [ns]'].values / 1e9
    pose_times = pose_data['#timestamp'].values / 1e9

    # Calibrate
    imu_calibrator = utils.getImuCalib("Euroc")
    b_g = imu_calibrator["gyro_bias"]
    b_a = imu_calibrator["accel_bias"]
    # w_calib = raw_imu[:, 1:4].T - b_g[:, None]
    # a_calib = raw_imu[:, 4:].T - b_a[:, None]

    # 插值 IMU 数据 (角速度和加速度)
    interp_gyro = interp1d(imu_times, imu_data[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].values, axis=0, fill_value="extrapolate")
    interp_accel = interp1d(imu_times, imu_data[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].values, axis=0, fill_value="extrapolate")
    # 插值位姿数据 (位置)
    interp_position = interp1d(pose_times, pose_data[[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]']].values, axis=0, fill_value="extrapolate")
    # 插值位姿数据 (速度)
    interp_velocity = interp1d(pose_times, pose_data[[' v_RS_R_x [m s^-1]', ' v_RS_R_y [m s^-1]', ' v_RS_R_z [m s^-1]']].values, axis=0, fill_value="extrapolate")
    # 插值位姿数据 (姿态 - 四元数)
    rotations = Rotation.from_quat(pose_data[[' q_RS_x []', ' q_RS_y []', ' q_RS_z []', ' q_RS_w []']].values)
    slerp = Slerp(pose_times, rotations)

    for split, time_range in times_config.items():
        start_time, end_time = time_range
        # 限制插值时间范围在 pose 可插值范围内
        valid_start_time = max(start_time, pose_times[0])
        valid_end_time = min(end_time, pose_times[-1])

        # 创建严格时间序列
        # 安全构造 strict_times，不超过 valid_end_time
        num_steps = int(np.floor((valid_end_time - valid_start_time) / dt))
        strict_times = valid_start_time + dt * np.arange(num_steps)

        # 获取插值数据
        gyro_data = interp_gyro(strict_times)
        accel_data = interp_accel(strict_times)

        position_data = interp_position(strict_times)
        velocity_data = interp_velocity(strict_times)
        orientation_data = slerp(strict_times).as_quat()

        # 初始位置姿态速度信息
        traj_target_oris_from_imu = np.concatenate((position_data[:1, :], orientation_data[:1, :], velocity_data[:1, :]), axis=1)

        # 合并插值后的数据
        combined_data = np.hstack([
            strict_times.reshape(-1, 1),  # 时间戳
            gyro_data,  # 角速度
            accel_data,  # 加速度
            position_data,  # 位置
            orientation_data,  # 姿态（四元数）
            velocity_data  # 速度
        ])

        output_dir_split = os.path.join(output_dir, split)
        if not os.path.exists(output_dir_split):
            os.makedirs(output_dir_split)
        # 保存为 HDF5 文件
        hdf5_path = os.path.join(output_dir_split, "data.hdf5")
        with h5py.File(hdf5_path, "w") as hdf:
            hdf.create_dataset("ts", data=combined_data[:, 0])
            hdf.create_dataset("gyro_raw", data=combined_data[:, 1:4])
            hdf.create_dataset("accel_raw", data=combined_data[:, 4:7])
            hdf.create_dataset("gyro_calib", data=combined_data[:, 1:4])
            hdf.create_dataset("accel_calib", data=combined_data[:, 4:7])
            hdf.create_dataset("traj_target", data=combined_data[:, 7:17])
            hdf.create_dataset("traj_target_oris_from_imu", data=traj_target_oris_from_imu)
            # hdf.create_dataset("thrust", data=thrusts_train)
            # hdf.create_dataset("i_thrust", data=i_thrusts_train)
            hdf.create_dataset("gyro_bias", data=b_g)
            hdf.create_dataset("accel_bias", data=b_a)

        print(f"数据保存到 {hdf5_path}")

        # 如果需要保存为文本文件
        if save_txt:
            txt_path = os.path.join(output_dir_split, "stamped_groundtruth_imu.txt")
            np.savetxt(
                txt_path, np.concatenate((combined_data[:, :1], combined_data[:, 7:14]), axis=1), fmt="%.6f",
                header="ts x y z qx qy qz qw"
            )
            print(f"数据保存到 {txt_path}")

def process_pose_data_row(row):
    """
    对pose_data的每行数据做坐标转换
    """
    # TODO:转换速度是为了得到初始速度，但速度转换可能存在问题
    t_w1_b = np.array([row[' p_RS_R_x [m]'], row[' p_RS_R_y [m]'], row[' p_RS_R_z [m]']])
    v_w1_b = np.array([row[' v_RS_R_x [m s^-1]'], row[' v_RS_R_y [m s^-1]'], row[' v_RS_R_z [m s^-1]']])
    R_w1_b = Rotation.from_quat(
        np.array([row[' q_RS_x []'], row[' q_RS_y []'], row[' q_RS_z []'],row[' q_RS_w []']])).as_matrix()
    
    R_w2_b = R_w2_w1 @ R_w1_b
    t_w2_b = R_w2_w1 @ t_w1_b + t_w2_w1
    v_w2_b = R_w2_w1 @ v_w1_b
    q_w2_b = Rotation.from_matrix(R_w2_b).as_quat()
    
    row[' p_RS_R_x [m]'] = t_w2_b[0]
    row[' p_RS_R_y [m]'] = t_w2_b[1]
    row[' p_RS_R_z [m]'] = t_w2_b[2]
    row[' v_RS_R_x [m s^-1]'] = v_w2_b[0]
    row[' v_RS_R_y [m s^-1]'] = v_w2_b[1]
    row[' v_RS_R_z [m s^-1]'] = v_w2_b[2]
    row[' q_RS_x []'] = q_w2_b[0]
    row[' q_RS_y []'] = q_w2_b[1]
    row[' q_RS_z []'] = q_w2_b[2]
    row[' q_RS_w []'] = q_w2_b[3]

    return row

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

