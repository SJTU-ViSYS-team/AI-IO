import os
import argparse
import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

import utils

'''
    python src/learning/data_management/prepare_datasets/simulation.py --dataset_dir datasets/Simulation/ --data_list data_list.txt
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

# body, imu坐标系一致
# rotate from imu to body frame
R_b_i = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]])
t_b_i = np.array([0., 0., 0.])
# # world: NED to NWU
# R_w2_w1 = np.array([
#     [1., 0., 0.],
#     [0., -1., 0.],
#     [0., 0., -1.]
# ])
# world: NWU to NWU
R_w2_w1 = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
])
t_w2_w1 = np.array([0., 0., 0.])

# 示例的时间范围配置（自定义）
# train_times = {
#     'mav0': [1403636900.0, 1403637000.0],
#     'sequence2': [20.0, 30.0]
# }
# val_times = {
#     'mav0': [1403636880.0, 1403636900.0],
#     'sequence2': [30.0, 35.0]
# }
# test_times = {
#     'mav0': [1403636860.0, 1403636880.0],
#     'sequence2': [35.0, 40.0]
# }
train_times = {
    'sim1': [0.0, 120.0],
    'sim1_double': [0.0, 240.0],
    'sim_line_add_noise_200Hz': [0.0 + 1e5, 70.0 + 1e5],
    'sim_rectangle_3_1_1': [0.0 + 1e5, 120.0 + 1e5],
    'sim_rectangle_8_1_1': [0.0 + 1e5, 160.0 + 1e5],
    'sim_rectangle_add_noise': [0.0 + 1e5, 160.0 + 1e5],
    'sim_rectangle_add_noise_200Hz': [0.0 + 1e5, 160.0 + 1e5],
    'altitude_hold': [0.0 + 1e5, 70.0 + 1e5],
    'FPV_500Hz_1000s': [0.0 + 1e5, 800.0 + 1e5],
    'FPV_sim_hover': [0.0 + 1e5, 20.0 + 1e5],
    'FPV_sim_line': [0.0 + 1e5, 20.0 + 1e5]
    
}
val_times = {
    'sim1': [120.0, 140.0],
    'sim1_double': [240.0, 280.0],
    'sim_line_add_noise_200Hz': [70.0 + 1e5, 85.0 + 1e5],
    'sim_rectangle_3_1_1': [120.0 + 1e5, 160.0 + 1e5],
    'sim_rectangle_8_1_1': [160.0 + 1e5, 180.0 + 1e5],
    'sim_rectangle_add_noise': [160.0 + 1e5, 180.0 + 1e5],
    'sim_rectangle_add_noise_200Hz': [160.0 + 1e5, 180.0 + 1e5],
    'altitude_hold': [70.0 + 1e5, 85.0 + 1e5],
    'FPV_500Hz_1000s': [800.0 + 1e5, 900.0 + 1e5],
    'FPV_sim_hover': [0.0 + 1e5, 20.0 + 1e5],
    'FPV_sim_line': [0.0 + 1e5, 20.0 + 1e5]
}
test_times = {
    'sim1': [140.0, 160.0],
    'sim1_double': [280.0, 320.0],
    'sim_line_add_noise_200Hz': [85.0 + 1e5, 100.0 + 1e5],
    'sim_rectangle_3_1_1': [160.0 + 1e5, 200.0 + 1e5],
    'sim_rectangle_8_1_1': [180.0 + 1e5, 200.0 + 1e5],
    'sim_rectangle_add_noise': [180.0 + 1e5, 200.0 + 1e5],
    'sim_rectangle_add_noise_200Hz': [180.0 + 1e5, 200.0 + 1e5],
    'altitude_hold': [85.0 + 1e5, 100.0 + 1e5],
    'FPV_500Hz_1000s': [900.0 + 1e5, 1000.0 + 1e5],
    'FPV_sim_hover': [0.0 + 1e5, 20.0 + 1e5],
    'FPV_sim_line': [0.0 + 1e5, 20.0 + 1e5]
}
dt = 0.002

def prepare_dataset(args):
    """
    处理数据集，生成 HDF5 数据集并根据时间范围划分训练/验证/测试集。
    """
    dataset_dir = args.dataset_dir

    # 获取数据序列文件路径
    with open(os.path.join(dataset_dir, args.data_list), 'r') as f:
        seq_names = [line.strip() for line in f.readlines()]

    # 遍历所有序列
    for seq_name in seq_names:
        seq_dir = os.path.join(dataset_dir, seq_name)
        imu_csv = os.path.join(seq_dir, "imu_data.csv")
        pose_csv = os.path.join(seq_dir, "pose_data.csv")

        # 确保文件存在
        assert os.path.isfile(imu_csv), f"IMU 数据文件缺失：{imu_csv}"
        assert os.path.isfile(pose_csv), f"位姿数据文件缺失：{pose_csv}"

        # 根据配置中的时间范围生成训练、验证、测试集
        times_config = {
            "train": train_times.get(seq_name),  # 示例默认时间
            "val": val_times.get(seq_name),
            "test": test_times.get(seq_name)
        }

        # 输出目录
        output_dir = os.path.join(seq_dir, "processed_data")

        # 调用数据处理函数
        process_and_save_to_hdf5(
            imu_csv=imu_csv,
            pose_csv=pose_csv,
            output_dir=output_dir,
            sequence_name=seq_name,
            times_config=times_config,
            save_txt=args.save_txt
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
        # 过滤时间范围内的数据
        start_time, end_time = time_range
        imu_mask = (imu_times >= start_time) & (imu_times <= end_time)
        imu_times_part = imu_times[imu_mask]
        # pose_mask = (pose_times >= start_time) & (pose_times <= end_time)

        # imu_data_part = imu_data[imu_mask]
        # pose_data_part = pose_data[pose_mask]
        # imu_times_part = imu_data_part['#timestamp [ns]'].values
        # pose_times_part = pose_data_part['#timestamp'].values

        # 生成严格的时间序列
        strict_times = np.arange(imu_times_part[0], imu_times_part[-1], dt)  # 0.005s 间隔

        # 获取插值数据
        gyro_data = interp_gyro(strict_times)
        accel_data = interp_accel(strict_times)

        position_data = interp_position(strict_times)
        velocity_data = interp_velocity(strict_times)
        orientation_data = slerp(strict_times).as_quat()

        # FIXME: 当前初始速度为b系速度，需要使用w系
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
                header="# ts x y z qx qy qz qw"
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
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="处理数据集并生成 HDF5 数据")
    parser.add_argument("--dataset_dir", type=str, required=True, help="数据集的主目录")
    parser.add_argument("--data_list", type=str, required=True, help="数据列表文件，包含序列名称")
    parser.add_argument("--save_txt", action="store_true", help="是否保存为文本文件", default=True)
    args = parser.parse_args()

    # 运行数据集准备函数
    prepare_dataset(args)
