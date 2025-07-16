"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""


import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from learning.utils import pose
from learning.utils.plot_utils import xyztPlot
import src.utils.plotting as plotting

from pyhocon import ConfigFactory

def process_sequence(seq_name, seq_path, dataset_name, result_dir):
    # seq = os.path.basename(os.path.dirname(os.path.dirname(seq_path)))  # 获取序列名（如 MH_01）
    print(f"Processing sequence: {seq_name}")
    
    out_dir = os.path.join(result_dir, dataset_name, seq_name, 'pyfilter')
    traj_fn = os.path.join(out_dir, "stamped_traj_estimate.txt")
    bias_fn = os.path.join(out_dir, "stamped_bias_estimate.txt")
    vel_fn = os.path.join(out_dir, "stamped_vel_estimate.txt")

    if not (os.path.exists(traj_fn) and os.path.exists(bias_fn) and os.path.exists(vel_fn)):
        print(f"Missing files for sequence {seq_name}. Skipping.")
        return

    traj = np.loadtxt(traj_fn)
    bias = np.loadtxt(bias_fn)
    ts = bias[:, 0]
    bg = bias[:, 1:4]
    ba = bias[:, 4:]
    vel = np.loadtxt(vel_fn)

    # dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(seq_path)))  # 回到 root
    gt_fn = os.path.join(seq_path, 'stamped_groundtruth_imu.txt')
    if not os.path.exists(gt_fn):
        print(f"Groundtruth file not found: {gt_fn}")
        return
    gt_traj = np.loadtxt(gt_fn)
    gt_ts = gt_traj[:, 0]

    # 时间对齐
    if traj[0, 0] < gt_ts[0]:
        idxs = np.argwhere(traj[:, 0] > gt_ts[0])[0][0]
    else:
        idxs = 0
    if traj[-1, 0] > gt_ts[-1]:
        idxe = np.argwhere(traj[:, 0] > gt_ts[-1])[0][0]
    else:
        idxe = traj.shape[0]
    traj = traj[idxs:idxe]

    gt_pos_data = interp1d(gt_traj[:, 0], gt_traj[:, 1:4], axis=0)(traj[:, 0])
    gt_rot_data = Slerp(gt_traj[:, 0], Rotation.from_quat(gt_traj[:, 4:8]))(traj[:, 0])
    gt_ts = traj[:, 0]
    gt_traj_interp = np.concatenate((gt_ts.reshape((-1, 1)), gt_pos_data, gt_rot_data.as_quat()), axis=1)

    # 估计速度
    dts = (gt_ts[2:] - gt_ts[:-2])
    vel_gt = (gt_traj_interp[2:, 1:4] - gt_traj_interp[:-2, 1:4]) / dts.reshape((-1, 1))
    vel_gt = np.concatenate((gt_ts[1:-1].reshape((-1, 1)), vel_gt), axis=1)

    ypr_gt = np.array([pose.fromQuatToEulerAng(targ[4:8]) for targ in gt_traj_interp])
    ypr_est = np.array([pose.fromQuatToEulerAng(targ[4:8]) for targ in traj])
    atti_gt = np.concatenate((gt_ts.reshape((-1, 1)), ypr_gt), axis=1)
    atti_est = np.concatenate((gt_ts.reshape((-1, 1)), ypr_est), axis=1)

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    # 绘图
    # xyz time plots
    plt.figure('XYZt view')
    xyztPlot('Position', traj[:,:4], 'estim. traj', gt_traj_interp[:,:4], 'gt')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "position.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "position.png"))
    plt.close()
    plotting.make_position_plots(traj, gt_traj_interp)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "trajectory.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "trajectory.png"))
    plt.close()
    plotting.plotBiases(ts, bg, ba)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "bias.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "bias.png"))
    plt.close()
    plotting.make_velocity_plots(vel, vel_gt)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "velocity.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "velocity.png"))
    plt.close()
    plotting.make_ori_euler_plots(atti_est, atti_gt)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "attitude.svg"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, "attitude.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, help="Path to data config")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    conf = ConfigFactory.parse_file(args.data_config)
    config = conf["test"]

    data_list = []
    for entry in config["data_list"]:
        root = entry["data_root"]
        drives = entry["data_drive"]
        for drive in drives:
            path = os.path.join(root, drive, "processed_data", config["mode"])
            data_list.append((drive, path))

    for seq_name, seq_path in data_list:
        try:
            process_sequence(seq_name, seq_path, args.dataset, args.result_dir)
        except Exception as e:
            print(f"Error processing {seq_path}: {e}")

    plt.show()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--data_config", type=str, help="Path to data config")
#     # parser.add_argument("--seq", type=str, required=True)
#     # parser.add_argument("--dataset_dir", type=str, required=True)
#     parser.add_argument("--result_dir", type=str, required=True)
#     parser.add_argument("--dataset", type=str, required=True)

#     args = parser.parse_args()

#     conf = ConfigFactory.parse_file(args.data_config)
#     config = conf["test"]
#     data_list = []
#     for entry in config["data_list"]:
#         root = entry["data_root"]
#         drives = entry["data_drive"]
#         for drive in drives:
#             data_list.append(os.path.join(root, drive, "processed_data", config["mode"]))

#     # dataset_dir = args.dataset_dir
#     result_dir = args.result_dir
#     out_dir = os.path.join(result_dir, args.dataset, args.seq, 'pyfilter')

#     # load data
#     traj_fn = os.path.join(out_dir, "stamped_traj_estimate.txt")
#     traj = np.loadtxt(traj_fn)

#     bias_fn = os.path.join(out_dir, "stamped_bias_estimate.txt")
#     bias = np.loadtxt(bias_fn)
#     ts = bias[:,0]
#     bg = bias[:,1:4]
#     ba = bias[:,4:]

#     vel_fn = os.path.join(out_dir, "stamped_vel_estimate.txt")
#     vel = np.loadtxt(vel_fn)

#     gt_fn = os.path.join(dataset_dir, args.dataset, args.seq, 'stamped_groundtruth_imu.txt')
#     gt_traj = np.loadtxt(gt_fn)
#     gt_ts = gt_traj[:, 0]

#     # sample at same times
#     if traj[0,0] < gt_ts[0]:
#         idxs = np.argwhere(traj[:, 0] > gt_ts[0])[0][0]
#     else:
#         idxs = 0
#     if traj[-1,0] > gt_ts[-1]:
#         idxe = np.argwhere(traj[:, 0] > gt_ts[-1])[0][0]
#     else:
#         idxe = traj.shape[0]
#     traj = traj[idxs:idxe]

#     gt_pos_data = interp1d(gt_traj[:,0], gt_traj[:,1:4], axis=0)(traj[:,0])
#     gt_rot_data = Slerp(gt_traj[:,0], Rotation.from_quat(gt_traj[:,4:8]))(traj[:,0])
#     gt_ts = traj[:,0]
#     gt_traj = np.concatenate((
#         gt_ts.reshape((-1,1)), gt_pos_data, gt_rot_data.as_quat()), axis=1)

#     # get estimate of velocity
#     dts = (gt_ts[2:] - gt_ts[:-2])
#     vel_gt = (gt_traj[2:, 1:4] - gt_traj[:-2, 1:4]) / dts.reshape((-1,1))
#     vel_gt = np.concatenate((gt_ts[1:-1].reshape((-1,1)), vel_gt), axis=1)

#     # get attitude angle
#     ypr_gt   = np.array([pose.fromQuatToEulerAng(targ[4:8]) for targ in gt_traj])
#     ypr_est  = np.array([pose.fromQuatToEulerAng(targ[4:8]) for targ in traj])
#     atti_gt  = np.concatenate((gt_ts.reshape((-1,1)), ypr_gt), axis=1)
#     atti_est = np.concatenate((gt_ts.reshape((-1,1)), ypr_est), axis=1)

#     # make plots
#     plotting.make_position_plots(traj, gt_traj)
#     plotting.plotBiases(ts, bg, ba)
#     plotting.make_velocity_plots(vel, vel_gt)
#     plotting.make_ori_euler_plots(atti_est, atti_gt)

#     plt.show()

