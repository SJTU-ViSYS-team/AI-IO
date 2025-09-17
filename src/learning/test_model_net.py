"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/network/test.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from learning.data_management.datasets import *
from learning.network.losses import get_loss
from learning.network.model_factory import get_model

from learning.utils.argparse_utils import arg_conversion
from learning.utils.logging import logging

from pyhocon import ConfigFactory
from learning.utils.error_analyze import *

def integrate_trajectory(ts, pred_vel_body, gt_traj):
    """
    convert veclocity predictions from body to world frame, integrate to trajectory
    ts: [N, window_size]
    pred_vel_body: [N, 3]
    gt_traj: [N, 7] or [N, 10]

    return: pred_pos_world [N, 3]
    """
    # Init
    pred_pos_world = np.zeros((pred_vel_body.shape[0], 3))
    pred_vel_world = np.zeros((pred_vel_body.shape[0], 3))
    pred_pos_world[0] = gt_traj[0, :3]

    time = ts[:, -1]
    dt = np.diff(time, prepend=time[0])

    for i in range(1, len(time)):
        # get quaternion (qx, qy, qz, qw)
        R = pose.xyzwQuatToMat(gt_traj[i, 3:7])

        # body vel -> world vel
        v_body = pred_vel_body[i]
        v_world = R @ v_body

        # integrate
        pred_vel_world[i] = v_world
        pred_pos_world[i] = pred_pos_world[i-1] + v_world * dt[i]

    return pred_pos_world, pred_vel_world

def makeErrorPlot(dp_errors, dp_cov):
    fig1 = plt.figure("Errors")
    gs = gridspec.GridSpec(3, 1)

    fig1.add_subplot(gs[0, 0])
    plt.plot(dp_errors[:,0], label='x')
    plt.grid()
    plt.legend()
    plt.xlabel('#')
    plt.ylabel('$[m]$')
    plt.title('Position errors')

    fig1.add_subplot(gs[1, 0])
    plt.plot(dp_errors[:,1], label='y')
    plt.grid()
    plt.legend()
    plt.xlabel('#')
    plt.ylabel('$[m]$')

    fig1.add_subplot(gs[2, 0])
    plt.plot(dp_errors[:,2], label='z')
    plt.grid()
    plt.legend()
    plt.xlabel('#')
    plt.ylabel('$[m]$')

    fig2 = plt.figure("Std")
    gs = gridspec.GridSpec(3, 1)

    fig2.add_subplot(gs[0, 0])
    plt.plot(np.sqrt(dp_cov[:,0]), label='x')
    plt.grid()
    plt.legend()
    plt.xlabel('#')
    plt.ylabel('$[m]$')
    plt.title('Position std')

    fig2.add_subplot(gs[1, 0])
    plt.plot(np.sqrt(dp_cov[:,1]), label='y')
    plt.grid()
    plt.legend()
    plt.xlabel('#')
    plt.ylabel('$[m]$')

    fig2.add_subplot(gs[2, 0])
    plt.plot(np.sqrt(dp_cov[:,2]), label='z')
    plt.grid()
    plt.legend()
    plt.xlabel('#')
    plt.ylabel('$[m]$')


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(learn_configs, network, data_loader, device, epoch):
    """
    Get network status
    """
    ts_all, targets_all = [], []
    pred_all, pred_cov_all = [], []
    errs_all, losses_all = [], []
    traj_all = []
    
    network.eval()

    for _, (feat, targ, gt_traj, ts, _, _) in enumerate(data_loader):
        # feat_i = [[acc], [gyro], [rotor speed], [6d rotation matrix]]
        # dims = [batch size, 16, window size]
        # targ = [dv]
        # dims = [batch size, 3]
        feat = feat.to(device)
        targ = targ.to(device)
        gt_traj = gt_traj.to(device)
        
        pred, pred_cov = network(feat)

        # compute loss
        loss = get_loss(pred, pred_cov, targ, epoch, learn_configs)
        errs = pred - targ
        
        # log
        losses_all.append(torch_to_numpy(loss))
        errs_all.append(torch_to_numpy(errs))
        ts_all.append(torch_to_numpy(ts))
        targets_all.append(torch_to_numpy(targ))
        pred_all.append(torch_to_numpy(pred))
        pred_cov_all.append(torch_to_numpy(pred_cov))
        traj_all.append(torch_to_numpy(gt_traj[:, -1, :]))

    losses_all = np.concatenate(losses_all, axis=0)
    errs_all = np.concatenate(errs_all, axis=0)
    errs_norm = np.linalg.norm(errs_all, axis=1)
    rmse = np.sqrt(np.mean(errs_all ** 2, axis=0))
    ts_all = np.concatenate(ts_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    pred_cov_all = np.concatenate(pred_cov_all, axis=0)
    traj_all = np.concatenate(traj_all, axis=0)
        
    attr_dict = {
        "losses": losses_all,
        "errs": errs_norm,
        "rmse": rmse,
        "ts": ts_all,
        "targets": targets_all,
        "pred_all": pred_all,
        "pred_cov_all": pred_cov_all,
        "traj_all": traj_all
        }

    return attr_dict


def get_datalist(config):
    data_list = []
    for entry in config["data_list"]:
        root = entry["data_root"]
        drives = entry["data_drive"]
        for drive in drives:
            data_list.append((drive, os.path.join(root, drive, "processed_data", config["mode"])))
    return data_list

def test(args):
    try:
        if args.data_config is None:
            raise ValueError("data_config must be specified.")

        conf = ConfigFactory.parse_file(args.data_config)

        if args.dataset is None:
            raise ValueError("dataset must be specified.")
        if args.out_dir is not None:
            if not os.path.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            logging.info(f"Testing output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    test_config = conf["test"]
    test_list = get_datalist(test_config)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model_path = os.path.join(args.out_dir, args.dataset, "checkpoints", "model_net", args.model_fn)
    checkpoint = torch.load(model_path, map_location=device)

    network = get_model(data_window_config["window_size"]).to(
        device
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    logging.info(f"Model {model_path} loaded to device {device}.")

    all_errors = []
    all_results = []
    # process sequences
    for seq_name, data in test_list:
        logging.info(f"Processing {seq_name}...")
        try:
            seq_dataset = construct_dataset(args, [data], data_window_config, mode="test")
            seq_loader = DataLoader(seq_dataset, batch_size=128, shuffle=False)
        except OSError as e:
            print(e)
            continue

        # Obtain outputs
        net_attr_dict = get_inference(net_config, network, seq_loader, device, args.epochs)

        # Print loss infos
        errs_vel = np.mean(net_attr_dict["errs"])
        loss = np.mean(net_attr_dict["losses"])
        
        logging.info(f"Test: average vel err [m/s]: {errs_vel}")
        logging.info(f"Test: average loss: {loss}")
            
        # save displacement related quantities
        ts = net_attr_dict["ts"]
        vel_targets = net_attr_dict["targets"]
        pred = net_attr_dict["pred_all"] # n*3
        gt_traj = net_attr_dict["traj_all"]
        pred_cov = net_attr_dict["pred_cov_all"]
        pred_cov[pred_cov<-4] = -4
        for i in range(3):
            pred_cov[:, i] = torch.exp(2 * torch.tensor(pred_cov[:, i], dtype=torch.float32))

        outdir = os.path.join(args.out_dir, args.dataset, seq_name)
        if os.path.exists(outdir) is False:
            os.makedirs(outdir)
        # save loss
        outfile = os.path.join(outdir, "net_losses.txt")
        np.savetxt(outfile, net_attr_dict["losses"])

        df = pd.DataFrame({
                "gt_vel_x": vel_targets[::5, 0],
                "gt_vel_y": vel_targets[::5, 1],
                "gt_vel_z": vel_targets[::5, 2],
                "pred_vel_x": pred[::5, 0],
                "pred_vel_y": pred[::5, 1],
                "pred_vel_z": pred[::5, 2],
        })

        csv_path = os.path.join(outdir, "net_prediction.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved net predictions to {csv_path}")

        # --- Trajectory Reconstruction ---
        pred_vel_body = pred  # [N, 3]
        pred_pos_world, pred_vel_world = integrate_trajectory(ts, pred_vel_body, gt_traj)

        gt_pos_world = gt_traj[:, :3] 
        gt_vel_world = gt_traj[:, 7:10]

        df = pd.DataFrame({
            "gt_pos_x": gt_pos_world[::5, 0],
            "gt_pos_y": gt_pos_world[::5, 1],
            "gt_pos_z": gt_pos_world[::5, 2],
            "pred_pos_x": pred_pos_world[::5, 0],
            "pred_pos_y": pred_pos_world[::5, 1],
            "pred_pos_z": pred_pos_world[::5, 2],
        })

        df_v = pd.DataFrame({
            "gt_vel_x": gt_vel_world[::5, 0],
            "gt_vel_y": gt_vel_world[::5, 1],
            "gt_vel_z": gt_vel_world[::5, 2],
            "pred_vel_x": pred_vel_world[::5, 0],
            "pred_vel_y": pred_vel_world[::5, 1],
            "pred_vel_z": pred_vel_world[::5, 2],
        })

        csv_path = os.path.join(outdir, "net_inf_pos.csv")
        vel_path = os.path.join(outdir, "net_inf_vel.csv")
        df.to_csv(csv_path, index=False)
        df_v.to_csv(vel_path, index=False)
        print(f"Saved net inference position and velocity to {outdir}")

        errors = compute_position_velocity_errors(
            est_pos=pred_pos_world, gt_pos=gt_pos_world,
            est_vel=pred_vel_world, gt_vel=gt_vel_world
        )
        row = {
                "ATE_our": errors['position_rmse'][-1],
                "AVE_our": errors['velocity_rmse'][-1],
                "RTE_our": errors['position_rrmse'][-1],
                "RVE_our": errors['velocity_rrmse'][-1],
            }
        all_results.append(row)

        # plotting
        if args.show_plots:
            plot_dir = os.path.join(outdir, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            # compute errors
            vel_errs = pred - vel_targets
            sum_vel = np.linalg.norm(pred, axis=1)
            plt.figure('Sum Speed')
    
            plt.plot(sum_vel)
            plt.xlabel("epoch")
            plt.ylabel('x(m/s)')

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "speed.svg"), bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir, "speed.png"))
            plt.close()

            # --- Velocity Plot ---
            plt.figure(figsize=(12, 6))
            for i, axis in enumerate(['x', 'y', 'z']):
                plt.subplot(3, 1, i+1)
                plt.plot(vel_targets[:, i], label=f'GT vel {axis}', color='black')
                plt.plot(pred[:, i], label=f'Net vel {axis}', linestyle='--')
                plt.ylabel(f'vel_{axis} [m/s]')
                plt.legend()
                plt.grid(True)
            plt.xlabel('Time [s]')
            plt.suptitle('Velocity Comparison (GT vs Network)')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "velocity.svg"), bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir, "velocity.png"))
            plt.close()
            
            # --- Errors Plot ---
            plt.figure('Errors')
            plt.subplot(3, 1, 1)
            plt.title("Errors")
            plt.plot(vel_errs[:, 0])
            plt.xlabel("epoch")
            plt.ylabel('x(m/s)')

            plt.subplot(3, 1, 2)
            plt.plot(vel_errs[:, 1])
            plt.xlabel("epoch")
            plt.ylabel('y(m/s)')

            plt.subplot(3, 1, 3)
            plt.plot(vel_errs[:, 2])
            plt.xlabel("epoch")
            plt.ylabel('z(m/s)')

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "velocity_errors.svg"), bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir, "velocity_errors.png"))
            plt.close()

            # --- Std Deviation Plot ---
            plt.figure('Std')
            plt.subplot(3, 1, 1)
            plt.title("Std Deviation")
            plt.plot(pred_cov[:, 0])
            plt.xlabel("epoch")
            plt.ylabel('x(m/s)')

            plt.subplot(3, 1, 2)
            plt.plot(pred_cov[:, 1])
            plt.xlabel("epoch")
            plt.ylabel('y(m/s)')

            plt.subplot(3, 1, 3)
            plt.plot(pred_cov[:, 2])
            plt.xlabel("epoch")
            plt.ylabel('z(m/s)')

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "velocity_std.svg"), bbox_inches='tight')
            plt.savefig(os.path.join(plot_dir, "velocity_std.png"))
            plt.close()

            with open(os.path.join(plot_dir, "error.txt"), 'w') as f:
                f.write("-- Vel Errors --\n")
                for i, axis in enumerate(['x', 'y', 'z']):
                    f.write(f'{axis}\n')
                    f.write('mean = %.5f\n' % np.mean(vel_errs[:, i]))
                    f.write('std = %.5f\n' % np.std(vel_errs[:, i]))
                    f.write('rmse = %.5f\n' % net_attr_dict["rmse"][i])

                f.write("-- Summary --\n")
                f.write(f"average vel err [m/s]: {errs_vel}\n")
                f.write(f"average loss: {loss}\n")

            all_errors.append(errs_vel)
        
    df = pd.DataFrame(all_results)
    df.to_csv("our2_inf_metrics.csv", index=False)
    print("Average Velocity Error on All Test Data: %.5f" % np.mean(np.array(all_errors)))
    
    return

def construct_dataset(args, data_list, data_window_config, mode="test"):
    if args.dataset == "DIDO":
        train_dataset = ModelDIDODataset(data_list, args, data_window_config, mode=mode)
    elif args.dataset == "our2":
        train_dataset = ModelOur2Dataset(data_list, args, data_window_config, mode=mode)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return train_dataset
