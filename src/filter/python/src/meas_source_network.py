"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/meas_source_network.py
"""

import numpy as np
import torch

from learning.network.model_factory import get_model
from filter.python.src.utils.logging import logging
from learning.network.covariance_parametrization import DiagonalParam

class MeasSourceNetwork:
    def __init__(self, model_path, force_cpu=False):
        # network
        self.net = get_model(6, 2)

        # load trained network model
        if not torch.cuda.is_available() or force_cpu:
            self.device = torch.device("cpu")
            checkpoint = torch.load(
                model_path, map_location=lambda storage, location: storage
            )
        else:
            self.device = torch.device("cuda:0")
            checkpoint = torch.load(model_path)

        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net.eval().to(self.device)
        logging.info("Model {} loaded to device {}.".format(model_path, self.device))

    def get_measurement(self, net_t_s, net_gyr_w, net_acc_w):
        meas, meas_cov = self.get_vb_measurement_model_net(
            net_t_s, net_gyr_w, net_acc_w)
        return meas, meas_cov

    def get_vb_measurement_model_net(self, net_t_s, net_gyr_w, net_accl_w):
        features = np.concatenate([net_gyr_w, net_accl_w], axis=1)  # N x 6
        features_t = torch.unsqueeze(
            torch.from_numpy(features.T).float().to(self.device), 0
        )  # 1 x 6 x N

        # get inference
        vb_learnt, vb_cov_learned = self.net(features_t)

        # define measurement
        meas = vb_learnt.cpu().detach().numpy()
        meas = meas.reshape((2, 1))
        # TODO: learn covariance
        # meas_cov = np.eye(3)
        vb_cov_learned[vb_cov_learned < -4] = -4  # exp(2 * -4) =~ 0.00034
        meas_cov = DiagonalParam.vec2Cov(vb_cov_learned).cpu().detach().numpy()[0, :, :]

        return meas, meas_cov

