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
        self.net = get_model(100)

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

    def get_measurement(self, net_t_s, net_accl_b, net_gyr_b, net_rotor):
        meas, meas_cov = self.get_vb_measurement_model_net(
            net_t_s, net_accl_b, net_gyr_b, net_rotor)
        return meas, meas_cov

    def get_vb_measurement_model_net(self, net_t_s, net_accl_b, net_gyr_b, net_rotor):
        features = np.concatenate([net_accl_b, net_gyr_b, net_rotor], axis=1)  # N x 10
        features_t = torch.unsqueeze(
            torch.from_numpy(features.T).float().to(self.device), 0
        )

        # get inference
        vb_learnt, vb_cov_learned = self.net(features_t)

        # define measurement
        meas = vb_learnt.cpu().detach().numpy()
        meas = meas.reshape((3, 1))

        vb_cov_learned[vb_cov_learned < -4] = -4  # exp(2 * -4) =~ 0.00034
        meas_cov = DiagonalParam.vec2Cov(vb_cov_learned).cpu().detach().numpy()[0, :, :]

        return meas, meas_cov

