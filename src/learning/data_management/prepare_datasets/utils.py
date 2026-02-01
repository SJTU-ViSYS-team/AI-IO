"""
Util functions for dataset preparation
Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/learning/data_management/prepare_datasets/utils.py
"""

import numpy as np


def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if (len(s.strip()) > 0 and not s.startswith("#"))]
    return data_list


def getImuCalib(dtset_name):
    init_calib = {}

    # DIDO
    init_calib["DIDO"] = {}
    init_calib["DIDO"]["gyro_bias"] = np.array([0.0, 0.0, 0.0])
    init_calib["DIDO"]["accel_bias"] = np.array([0.0, 0.0, 0.0])
    init_calib["DIDO"]["T_mat_gyro"] = np.eye(3)
    init_calib["DIDO"]["T_sens_gyro"] = np.zeros((3, 3))
    init_calib["DIDO"]["T_mat_accel"] = np.eye(3)
    # our2
    init_calib["our2"] = {}
    init_calib["our2"]["gyro_bias"] = np.array([0.0, 0.0, 0.0])
    init_calib["our2"]["accel_bias"] = np.array([0.0, 0.0, 0.0])
    init_calib["our2"]["T_mat_gyro"] = np.eye(3)
    init_calib["our2"]["T_sens_gyro"] = np.zeros((3, 3))
    init_calib["our2"]["T_mat_accel"] = np.eye(3)
    return init_calib[dtset_name]

