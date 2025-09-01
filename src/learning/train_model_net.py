"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/network/train.py
"""

import datetime
import json
import os
import signal
import sys
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.data_management.datasets import *
from learning.network.losses import get_error_and_loss, get_loss
from learning.network.model_factory import get_model
from learning.utils.argparse_utils import arg_conversion
from learning.utils.logging import logging
from learning.utils.visualize_utils import *

from pyhocon import ConfigFactory
from tqdm import tqdm

def get_datalist(config):
    data_list = []
    for entry in config["data_list"]:
        root = entry["data_root"]
        drives = entry["data_drive"]
        for drive in drives:
            data_list.append(os.path.join(root, drive, "processed_data", config["mode"]))
    return data_list


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(learn_configs, network, data_loader, device, epoch):
    """
    Get network status
    """
    errors_all, losses_all, preds_cov_all = [], [], []
    
    network.eval()

    pbar = tqdm(data_loader, ncols=100)
    for (feat, targ, gt_traj, ts, gyro, accel) in pbar:
        # feat_i = [[acc], [gyro], [rotor speed], [6d rotation matrix]]
        # dims = [batch size, 16, window size]
        # targ = [v]
        # dims = [batch size, 3]

        ts = ts.to(device).to(torch.float32)
        ts = ts - ts[:, 0:1]
        feat = feat.to(device)
        targ = targ.to(device)
        gt_traj = gt_traj.to(device)

        # get network prediction
        pred, pred_cov = network(feat, ts)

        # compute loss
        loss = get_loss(pred, pred_cov, targ, epoch, learn_configs)
        errs = pred - targ
        errs_norm = np.linalg.norm(torch_to_numpy(errs), axis=1)
        # log
        errors_all.append(errs_norm)
        losses_all.append(torch_to_numpy(loss))
        preds_cov_all.append(torch_to_numpy(pred_cov))

        pbar.set_description(f"loss: {np.mean(torch_to_numpy(loss)):.3f}, err: {np.mean(errs_norm):.3f}")
        
    # save
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    errors_all = np.concatenate(errors_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)

    # errors_all = np.concatenate(errors_all, axis=0)

    attr_dict = {
        "preds_cov":preds_cov_all,
        "errors": errors_all,
        "losses": losses_all
    }
    
    return attr_dict


def run_train(learn_configs, network, train_loader, device, optimizer, epoch):
    """
    Train network for one epoch
    """
    errors_all, losses_all, preds_cov_all = [], [], []
    # errors_all, losses_all = [], []

    network.train()

    for _, (feat, targ, gt_traj, ts, gyro, accel) in enumerate(train_loader):
        # feat_i = [[acc], [gyro], [rotor speed], [6d rotation matrix]]
        # dims = [batch size, 16, window size]
        # targ = [v]
        # dims = [batch size, 3]
        ts = ts.to(device).to(torch.float32)
        ts = ts - ts[:, 0:1]
        feat = feat.to(device)
        targ = targ.to(device)
        gt_traj = gt_traj.to(device)

        optimizer.zero_grad()

        # get network prediction
        pred, pred_cov = network(feat, ts)

        # compute loss
        loss = get_loss(pred, pred_cov, targ, epoch, learn_configs)
        errs = pred - targ
        errs_norm = np.linalg.norm(torch_to_numpy(errs), axis=1)
        # log
        errors_all.append(errs_norm)
        losses_all.append(torch_to_numpy(loss))
        preds_cov_all.append(torch_to_numpy(pred_cov))

        # backprop and optimization
        loss = loss.mean()
        loss.backward()
        optimizer.step()

    # save
    errors_all = np.concatenate(errors_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)

    train_dict = {
        "preds_cov":preds_cov_all,
        "errors": errors_all,
        "losses": losses_all
    }

    return train_dict


def write_summary(summary_writer, attr_dict, epoch, optimizer, mode):
    """ Given the attr_dict write summary and log the losses """
    error = np.mean(attr_dict["errors"])
    loss = np.mean(attr_dict["losses"])
    summary_writer.add_scalar(f"{mode}_loss_vel/avg", error, epoch)
    summary_writer.add_scalar(f"{mode}_loss/avg", loss, epoch)
    logging.info(f"{mode}: average error [m/s]: {error}")
    logging.info(f"{mode}: average loss: {loss}")

    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1)


def save_model(args, epoch, network, optimizer, interrupt=False):
    if interrupt:
        model_path = os.path.join(args.out_dir, "checkpoints", "model_net", "checkpoint_latest.pt")
    else:
        model_path = os.path.join(args.out_dir, "checkpoints", "model_net", "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")


def train(args):
    try:
        if args.data_config is None:
            raise ValueError("data_config must be specified.")

        conf = ConfigFactory.parse_file(args.data_config)

        if args.dataset is None:
            raise ValueError("dataset must be specified.")
        args.out_dir = os.path.join(args.out_dir, args.dataset)
        if args.out_dir != None:
            if not os.path.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            if not os.path.isdir(os.path.join(args.out_dir, "checkpoints")):
                os.makedirs(os.path.join(args.out_dir, "checkpoints"))
            if not os.path.isdir(os.path.join(args.out_dir, "checkpoints", "model_net")):
                os.makedirs(os.path.join(args.out_dir, "checkpoints", "model_net"))
            if not os.path.isdir(os.path.join(args.out_dir, "logs")):
                os.makedirs(os.path.join(args.out_dir, "logs"))
            with open(
                os.path.join(args.out_dir, "checkpoints", "model_net", "model_net_parameters.json"), "w"
            ) as parameters_file:
                parameters_file.write(json.dumps(vars(args), sort_keys=True, indent=4))
            logging.info(f"Training output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")

        train_config = conf["train"]
        train_list = get_datalist(train_config)

        run_validation = True
        val_config = conf["val"]
        val_list = get_datalist(val_config)

        if args.continue_from != None:
            if os.path.exists(args.continue_from):
                logging.info(
                    f"Continue training from existing model {args.continue_from}"
                )
            else:
                raise ValueError(
                    f"continue_from model file path {args.continue_from} does not exist"
                )
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info("Training/testing with " + str(data_window_config["sampling_freq"]) + " Hz gyro / thrust data")
    logging.info(
        "Window time: " + str(args.window_time)
        + " [s], " 
        + "Window size: " + str(data_window_config["window_size"])
        + ", "
        + "Window shift time: " + str(data_window_config["window_shift_time"])
        + " [s], "
        + "Window shift size: " + str(data_window_config["window_shift_size"])
    )

    # Network
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )    
    input_dim = args.input_dim
    output_dim = args.output_dim
    network = get_model(input_dim, output_dim, data_window_config["window_size"]).to(
        device
    )

    n_params = network.get_num_params()
    params = network.parameters()
    logging.info(f'TCN network loaded to device {device}')
    logging.info(f"Total number of learning parameters: {n_params}")

    # Training / Validation datasets
    train_loader, val_loader = None, None
    start_t = time.time()
    try:
        train_dataset = construct_dataset(args, train_list, data_window_config)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True) #, num_workers=16)
    except OSError as e:
        logging.error(e)
        return
    end_t = time.time()
    logging.info(f"Training set loaded. Loading time: {end_t - start_t:.3f}s")
    logging.info(f"Number of train samples: {len(train_dataset)}")

    trainable_params = filter(lambda p: p.requires_grad, network.parameters())
    optimizer = torch.optim.Adam(trainable_params, args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, eps=1e-5
    )
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    start_epoch = 0
    if args.continue_from != None:
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get("epoch", 0)
        network.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")
    else:
        # default starting from latest checkpoint from interruption
        latest_pt = os.path.join(args.out_dir, "checkpoints", "inertial_net", "checkpoint_latest.pt")

        if os.path.isfile(latest_pt):
            checkpoints = torch.load(latest_pt)
            start_epoch = checkpoints.get("epoch", 0)
            network.load_state_dict(checkpoints.get("model_state_dict"))
            optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
            logging.info(
                f"Detected saved checkpoint, starting from epoch {start_epoch}"
            )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_writer = SummaryWriter(os.path.join(args.out_dir, "logs", timestamp))
    summary_writer.add_text("info", f"total_param: {n_params}")

    logging.info(f"-------------- Init, Epoch {start_epoch} --------------")
    attr_dict = get_inference(net_config, network, train_loader, device, start_epoch)
    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train")
    best_loss = np.mean(attr_dict["losses"])
    # run first validation of the full validation set
    if run_validation:
        try:
            val_dataset = construct_dataset(args, val_list, data_window_config, mode="val")
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)  # , num_workers=16)
        except OSError as e:
            logging.error(e)
            return
        logging.info("Validation set loaded.")
        logging.info(f"Number of val samples: {len(val_dataset)}")

        val_dict = get_inference(net_config, network, val_loader, device, start_epoch)
        write_summary(summary_writer, val_dict, start_epoch, optimizer, "val")
        best_loss = np.mean(val_dict["losses"])

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, interrupt=True)
        sys.exit()

    ##############################################
    ############ actual training loop ############
    ##############################################
    visualize_path = os.path.join(args.out_dir, "visualize", timestamp)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        signal.signal(
            signal.SIGINT, partial(stop_signal_handler, args, epoch, network, optimizer)
        )
        signal.signal(
            signal.SIGTERM,
            partial(stop_signal_handler, args, epoch, network, optimizer),
        )

        logging.info(f"-------------- Training, Epoch {epoch} ---------------")
        start_t = time.time()
        train_dict = run_train(net_config, network, train_loader, device, optimizer, epoch)
        write_summary(summary_writer, train_dict, epoch, optimizer, "train")
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")

        if run_validation:
            val_attr_dict = get_inference(net_config, network, val_loader, device, epoch)
            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val")
            current_loss = np.mean(val_attr_dict["losses"])
            if epoch % 5 == 0 and args.visualize_net:
                visualize_net(network, visualize_path, epoch)
            # scheduler.step(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                save_model(args, epoch, network, optimizer)
            if epoch % args.save_interval == 0:
                save_model(args, epoch, network, optimizer)
        else:
            attr_dict = get_inference(net_config, network, train_loader, device, epoch)
            current_loss = np.mean(attr_dict["losses"])
            if current_loss < best_loss:
                best_loss = current_loss
                save_model(args, epoch, network, optimizer)
            if epoch % args.save_interval == 0:
                save_model(args, epoch, network, optimizer)

    logging.info("Training complete.")

    return


def construct_dataset(args, data_list, data_window_config, mode="train"):
    if args.dataset == "Euroc":
        train_dataset = ModelEurocDataset(data_list, args, data_window_config, mode=mode)
    elif args.dataset == "DIDO":
        train_dataset = ModelDIDODataset(data_list, args, data_window_config, mode=mode)
    elif args.dataset == "Blackbird":
        train_dataset = ModelBlackbirdDataset(data_list, args, data_window_config, mode=mode)
    elif args.dataset == "FPV":
        train_dataset = ModelFPVDataset(data_list, args, data_window_config, mode=mode)
    elif args.dataset == "Simulation":
        train_dataset = ModelSimulationDataset(data_list, args, data_window_config, mode=mode)
    elif args.dataset == "our2":
        train_dataset = ModelOur2Dataset(data_list, args, data_window_config, mode=mode)
    elif args.dataset == "ours":
        train_dataset = ModelOursDataset(data_list, args, data_window_config, mode=mode)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return train_dataset