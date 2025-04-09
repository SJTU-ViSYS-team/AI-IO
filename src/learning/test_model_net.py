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
import torch
from torch.utils.data import DataLoader

from learning.data_management.datasets import *
from learning.network.losses import get_error_and_loss, get_loss
from learning.network.model_factory import get_model

from learning.utils.argparse_utils import arg_conversion
from learning.utils.logging import logging


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


def get_inference(learn_configs, network, data_loader, device, epoch, debias_net=None):
    """
    Get network status
    """
    ts_all, targets_all = [], []
    pred_all, pred_cov_all = [], []
    errs_all, losses_all = [], []
    
    network.eval()
    if debias_net:
        debias_net.eval()

    for _, (feat, targ, ts, _, _) in enumerate(data_loader):
        # feat_i = [[feat_gyros], [feat_thrusts]]
        # dims = [batch size, 6, window size]
        # targ = [dv]
        # dims = [batch size, 3]
        feat = feat.to(device)
        targ = targ.to(device)
        # 预处理加速度数据
        if debias_net:
            with torch.no_grad():
                accel_data = feat[:, 3:6, :]  # 取出加速度数据
                accel_debiased = accel_data + debias_net(accel_data).unsqueeze(2).repeat(1,1, accel_data.shape[2])  # 通过去偏网络
                feat[:, 3:6, :] = accel_debiased  # 替换原来的加速度数据
        
        pred, pred_cov = network(feat)

        # compute loss
        loss = get_loss(pred, pred_cov, targ, epoch, learn_configs)
        errs = pred - targ
        errs_norm = np.linalg.norm(torch_to_numpy(errs), axis=1)
        
        # log
        losses_all.append(torch_to_numpy(loss))
        # errs_norm = np.linalg.norm(torch_to_numpy(errs), axis=1)
        errs_all.append(errs_norm)

        ts_all.append(torch_to_numpy(ts))
        targets_all.append(torch_to_numpy(targ))

        pred_all.append(torch_to_numpy(pred))
        pred_cov_all.append(torch_to_numpy(pred_cov))

    losses_all = np.concatenate(losses_all, axis=0)
    errs_all = np.concatenate(errs_all, axis=0)

    ts_all = np.concatenate(ts_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    pred_all = np.concatenate(pred_all, axis=0)
    pred_cov_all = np.concatenate(pred_cov_all, axis=0)
        
    attr_dict = {
        "losses": losses_all,
        "errs": errs_all,
        "ts": ts_all,
        "targets": targets_all,
        "dp_learned": pred_all,
        "dp_cov_learned": pred_cov_all
        }

    return attr_dict


def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if (len(s.strip()) > 0 and not s.startswith("#"))]
    return data_list

def test(args):
    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.test_list is None:
            raise ValueError("test_list must be specified.")
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

    test_list = get_datalist(os.path.join(args.root_dir, args.dataset, args.test_list))
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model_path = os.path.join(args.out_dir, args.dataset, "checkpoints", "model_net", args.model_fn)
    checkpoint = torch.load(model_path, map_location=device)

    input_dim = args.input_dim
    output_dim = args.output_dim
    network = get_model(input_dim, output_dim).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    logging.info(f"Model {model_path} loaded to device {device}.")

    #TODO: add debias network
    if args.debias_accel:
        from debias.network.model_factory import get_model as get_debias_model
        debias_model_path = args.debias_model_path
        debias_checkpoint = torch.load(debias_model_path, map_location=device)
        debias_net_config = {
        "in_dim": (
            100
        )}
        debias_net = get_debias_model("resnet", debias_net_config, 3, 3).to(device)
        debias_net.load_state_dict(debias_checkpoint["model_state_dict"])
        debias_net.eval()
        logging.info(f"Model {args.debias_model_path} loaded to device {device}.")
    else:
        debias_net = None

    # process sequences
    for data in test_list:
        logging.info(f"Processing {data}...")
        try:
            seq_dataset = construct_dataset(args, [data], data_window_config, mode="test")
            seq_loader = DataLoader(seq_dataset, batch_size=128, shuffle=False)
        except OSError as e:
            print(e)
            continue

        # Obtain outputs
        net_attr_dict = get_inference(net_config, network, seq_loader, device, 50, debias_net)

        # Print loss infos
        print(net_attr_dict["errs"].shape)
        print(net_attr_dict["losses"].shape)
        errs_pos = np.mean(net_attr_dict["errs"])
        loss = np.mean(net_attr_dict["losses"])
        

        logging.info(f"Test: average err [m]: {errs_pos}")
        logging.info(f"Test: average loss: {loss}")
            
        # save displacement related quantities
        ts = net_attr_dict["ts"]
        dp_learned = net_attr_dict["dp_learned"] # n*3
        dp_learned_sampled = np.concatenate((ts[:, 0].reshape(-1, 1), ts[:, -1].reshape(-1, 1), dp_learned), axis=1)
        dp_cov_learned = net_attr_dict["dp_cov_learned"]
        dp_cov_learned[dp_cov_learned<-4] = -4
        for i in range(3):
            dp_cov_learned[:, i] = torch.exp(2 * torch.tensor(dp_cov_learned[:, i], dtype=torch.float32))
        dp_cov_learned_sampled = np.concatenate((ts[:, 0].reshape(-1, 1), ts[:, -1].reshape(-1, 1), dp_cov_learned), axis=1)


        outdir = os.path.join(args.out_dir, args.dataset, data)
        if os.path.exists(outdir) is False:
            os.makedirs(outdir)
        outfile = os.path.join(outdir, "model_net_learnt_predictions.txt")
        np.savetxt(outfile, dp_learned_sampled, fmt="%.12f", header="t0 t1 dpx dpy dpz")
        outfile = os.path.join(outdir, "model_net_learnt_predictions_covariance.txt")
        np.savetxt(outfile, dp_cov_learned_sampled, fmt="%.5f", header="t0 t1 covx covy covz")

        # save loss
        outfile = os.path.join(outdir, "net_losses.txt")
        np.savetxt(outfile, net_attr_dict["losses"])

        # plotting
        if args.show_plots:
            # compute errors
            dp_targets = net_attr_dict["targets"]
            dp_errs = dp_learned - dp_targets

            plt.figure('Velocity')
            plt.subplot(3, 1, 1)
            plt.plot(dp_learned[:, 0], label="Learned x")
            plt.plot(dp_targets[:, 0], label="Real x", alpha=0.7)
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(dp_learned[:, 1], label="Learned y")
            plt.plot(dp_targets[:, 1], label="Real y", alpha=0.7)
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(dp_learned[:, 2], label="Learned z")
            plt.plot(dp_targets[:, 2], label="Real z", alpha=0.7)
            plt.legend()

            plt.figure('Errors')
            plt.subplot(3, 1, 1)
            plt.plot(dp_errs[:, 0], label="x")
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(dp_errs[:, 1], label="y")
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(dp_errs[:, 2], label="z")
            plt.legend()

            plt.figure('Std')
            plt.subplot(3, 1, 1)
            plt.plot(dp_cov_learned[:, 0], label="x")
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(dp_cov_learned[:, 1], label="y")
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(dp_cov_learned[:, 2], label="z")
            plt.legend()

            # fig1 = plt.figure("Errors")
            # gs = gridspec.GridSpec(3, 1)

            # fig1.add_subplot(gs[0, 0])
            # plt.plot(dp_errors[:,0], label='x')
            # plt.grid()
            # plt.legend()
            # plt.xlabel('#')
            # plt.ylabel('$[m]$')
            # plt.title('Position errors')

            # fig1.add_subplot(gs[1, 0])
            # plt.plot(dp_errors[:,1], label='y')
            # plt.grid()
            # plt.legend()
            # plt.xlabel('#')
            # plt.ylabel('$[m]$')

            # fig1.add_subplot(gs[2, 0])
            # plt.plot(dp_errors[:,2], label='z')
            # plt.grid()
            # plt.legend()
            # plt.xlabel('#')
            # plt.ylabel('$[m]$')

            # fig2 = plt.figure("Std")
            # gs = gridspec.GridSpec(3, 1)

            # fig2.add_subplot(gs[0, 0])
            # plt.plot(np.sqrt(dp_cov[:,0]), label='x')
            # plt.grid()
            # plt.legend()
            # plt.xlabel('#')
            # plt.ylabel('$[m]$')
            # plt.title('Position std')

            # fig2.add_subplot(gs[1, 0])
            # plt.plot(np.sqrt(dp_cov[:,1]), label='y')
            # plt.grid()
            # plt.legend()
            # plt.xlabel('#')
            # plt.ylabel('$[m]$')

            # fig2.add_subplot(gs[2, 0])
            # plt.plot(np.sqrt(dp_cov[:,2]), label='z')
            # plt.grid()
            # plt.legend()
            # plt.xlabel('#')
            # plt.ylabel('$[m]$')

            # makeErrorPlot(dp_errs, dp_cov_learned)

            print("-- dp Errors --")
            print('x')
            print('mean = %.5f' % np.mean(dp_errs[:,0]))
            print('std = %.5f' % np.std(dp_errs[:,0]))
            print('y')
            print('mean = %.5f' % np.mean(dp_errs[:,1]))
            print('std = %.5f' % np.std(dp_errs[:,1]))
            print('z')
            print('mean = %.5f' % np.mean(dp_errs[:,2]))
            print('std = %.5f' % np.std(dp_errs[:,2]))

            if args.show_plots:
                plt.show()
    return

def construct_dataset(args, data_list, data_window_config, mode="train"):
    if args.dataset == "DIDO":
        train_dataset = ModelDIDODataset(
            args.root_dir, args.dataset, data_list, args, data_window_config, mode=mode)
    elif args.dataset == "Blackbird":
        train_dataset = ModelBlackbirdDataset(
            args.root_dir, args.dataset, data_list, args, data_window_config, mode=mode)
    elif args.dataset == "FPV":
        train_dataset = ModelFPVDataset(
            args.root_dir, args.dataset, data_list, args, data_window_config, mode=mode)
    elif args.dataset == "Simulation":
        train_dataset = ModelSimulationDataset(
            args.root_dir, args.dataset, data_list, args, data_window_config, mode=mode)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return train_dataset