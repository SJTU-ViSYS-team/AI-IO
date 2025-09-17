"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Main script. Use it to load args and launch network training / test.

Reference: https://github.com/CathIAS/TLIO/blob/master/src/main_net.py
"""

import learning.train_model_net as train_model_net
import learning.test_model_net as test_model_net
from learning.utils.argparse_utils import add_bool_arg

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)                
    np.random.seed(seed)             
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set as {seed}")

if __name__ == "__main__":
    set_seed(42)
    import argparse

    parser = argparse.ArgumentParser()

    # ------------------ dataset -----------------
    parser.add_argument("--data_config", type=str, help="Path to data config")
    parser.add_argument("--out_dir", type=str, help="Path to result directory")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_fn", type=str, default=None)
    parser.add_argument("--continue_from", type=str, default=None)

    # ------------------ architecture and learning params -----------------
    parser.add_argument("--lr", type=float, default=3e-04)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60, help="max num epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="save model every n epochs")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--weight_vel_err", type=float, default=1.)
    parser.add_argument("--loss_type", type=str, default="huber", help="huber,mse")
    parser.add_argument("--huber_vel_loss_delta", type=float, default=0.1, help="value is in [m/s]")
    parser.add_argument("--switch_iter", type=int, default=50, help="switch to optimize covariance after this iter")

    # ------------------ data perturbation ------------------
    add_bool_arg(parser, "perturb_orientation", default=True) # TODO: delete
    parser.add_argument(
        "--perturb_orientation_theta_range", type=float, default=5.0
    )  # degrees
    add_bool_arg(parser, "perturb_accel", default=False)
    parser.add_argument("--perturb_accel_range", type=float, default=0.1)  # m/s^2

    # ------------------ commons -----------------
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "eval"]
    )
    parser.add_argument("--imu_freq", type=float, help="imu freq [Hz]")
    parser.add_argument("--sampling_freq", type=float, default=-1.0,
                        help="freq of imu sampling [Hz]. (-1.0 = same as imu_freq)")
    parser.add_argument("--window_time", type=float, default=1, help="[s]")
    parser.add_argument("--window_shift_size", type=int, default=1,
                        help="shift size of the input data window")

    # ----- plotting and evaluation -----
    add_bool_arg(parser, "show_plots", default=True)

    args = parser.parse_args()

    ###########################################################
    # Main
    ###########################################################
    if args.mode == "train":
        train_model_net.train(args)
    elif args.mode == "test":
        test_model_net.test(args)
    else:
        raise ValueError("Undefined mode")

