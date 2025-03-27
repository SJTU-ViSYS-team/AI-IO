import os
import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
# from learning.data_management.prepare_datasets import utils
# from learning.utils import pose

def process_hdf5(input_path, output_path):
    """处理单个HDF5文件的函数"""
    with h5py.File(input_path, "r") as f:
        ts = np.copy(f["ts"])
        gyro_raw = np.copy(f["gyr"])
        accel_raw = np.copy(f["acc"])
        gt_p = np.copy(f["gt_p"])
        gt_q = np.copy(f["gt_q"])
        gt_q = np.hstack((gt_q[:, 1:], gt_q[:, 0:1]))
        gt_v = np.copy(f["gt_v"])
        gt_vb = np.array([Rotation.from_quat(q).as_matrix().T @ v for v, q in zip(gt_v, gt_q)])
        traj_target = np.hstack((gt_p, gt_q, gt_vb))

    strict_ts = np.arange(ts[0], ts[-1], 0.0025)
    strict_ts = np.clip(strict_ts, ts[0], ts[-1])  # 将超限时间截断到边界
    interp_gyro = interp1d(ts, gyro_raw, axis=0, fill_value="extrapolate")(strict_ts)
    interp_accel = interp1d(ts, accel_raw, axis=0, fill_value="extrapolate")(strict_ts)
    interp_pos = interp1d(ts, gt_p, axis=0, fill_value="extrapolate")(strict_ts)
    interp_vb = interp1d(ts, gt_vb, axis=0, fill_value="extrapolate")(strict_ts)
    interp_quat = Slerp(ts, Rotation.from_quat(gt_q))(strict_ts).as_quat()

    traj_target = np.hstack((interp_pos, interp_quat, interp_vb))
    traj_target_oris_from_imu = np.concatenate((gt_p[:1, :], gt_q[:1, :], gt_v[:1, :]), axis=1)

    # imu_calibrator = utils.getImuCalib("Euroc")
    # b_g = imu_calibrator["gyro_bias"]
    # b_a = imu_calibrator["accel_bias"]
    b_g = np.zeros((3,1))
    b_a = np.zeros((3,1))

    with h5py.File(output_path, "w") as hdf:
        hdf.create_dataset("ts", data=strict_ts)
        hdf.create_dataset("gyro_raw", data=interp_gyro)
        hdf.create_dataset("accel_raw", data=interp_accel)
        hdf.create_dataset("gyro_calib", data=interp_gyro)
        hdf.create_dataset("accel_calib", data=interp_accel)
        hdf.create_dataset("traj_target", data=traj_target)
        hdf.create_dataset("traj_target_oris_from_imu", data=traj_target_oris_from_imu)
        hdf.create_dataset("gyro_bias", data=b_g)
        hdf.create_dataset("accel_bias", data=b_a)

    txt_path = os.path.join(os.path.dirname(input_path), "stamped_groundtruth_imu.txt")
    np.savetxt(
                txt_path, np.concatenate((strict_ts.reshape((-1, 1)), interp_pos, interp_quat), axis=1), fmt="%.6f",
                header="# ts x y z qx qy qz qw"
            )
    
def save_groundtruth(input_path):
    with h5py.File(input_path, "r") as f:
        ts = np.copy(f["ts"])
        gyro_raw = np.copy(f["gyr"])
        accel_raw = np.copy(f["acc"])
        gt_p = np.copy(f["gt_p"])
        gt_q = np.copy(f["gt_q"])
        gt_q = np.hstack((gt_q[:, 1:], gt_q[:, 0:1]))
        gt_v = np.copy(f["gt_v"])
        gt_vb = np.array([Rotation.from_quat(q).as_matrix().T @ v for v, q in zip(gt_v, gt_q)])
        traj_target = np.hstack((gt_p, gt_q, gt_vb))

    strict_ts = np.arange(ts[0], ts[-1], 0.0025)
    strict_ts = np.clip(strict_ts, ts[0], ts[-1])  # 将超限时间截断到边界
    interp_gyro = interp1d(ts, gyro_raw, axis=0, fill_value="extrapolate")(strict_ts)
    interp_accel = interp1d(ts, accel_raw, axis=0, fill_value="extrapolate")(strict_ts)
    interp_pos = interp1d(ts, gt_p, axis=0, fill_value="extrapolate")(strict_ts)
    interp_vb = interp1d(ts, gt_vb, axis=0, fill_value="extrapolate")(strict_ts)
    interp_quat = Slerp(ts, Rotation.from_quat(gt_q))(strict_ts).as_quat()

    txt_path = os.path.join(os.path.dirname(input_path), "stamped_groundtruth_imu.txt")
    np.savetxt(
                txt_path, np.concatenate((strict_ts.reshape((-1, 1)), interp_pos, interp_quat), axis=1), fmt="%.6f",
                header="# ts x y z qx qy qz qw"
            )

def batch_process(root_dir):
    """批量处理函数"""
    for subdir in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(dir_path):
            original_file = os.path.join(dir_path, "data.hdf5")
            backup_file = os.path.join(dir_path, "data_original.hdf5")
            new_file = os.path.join(dir_path, "data.hdf5")
            
            if os.path.exists(original_file) and not os.path.exists(backup_file):
                # 重命名原文件
                os.rename(original_file, backup_file)
                print(f"Processing: {dir_path}")
                try:
                    # 处理文件
                    process_hdf5(backup_file, new_file)
                    print(f"Success: {dir_path}")
                except Exception as e:
                    print(f"Error processing {dir_path}: {str(e)}")
                    # 恢复原文件名
                    os.rename(backup_file, original_file)
            elif os.path.exists(original_file) and os.path.exists(backup_file):
                save_groundtruth(backup_file)
            else:
                print(f"Skipping {dir_path}: no data.hdf5 found")

batch_process("./datasets/DIDO_temp")