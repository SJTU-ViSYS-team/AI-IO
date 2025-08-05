"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/network/model_factory.py
"""

from learning.network.model_tcn import Tcn, TcnGru, IMUTransformerWithModality


def get_model(input_dim=9, output_dim=3, window_s=50):
    network = IMUTransformerWithModality(
        sub_dim=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        output_size=3,
        window_size=window_s,
        enabled_modalities=["acc", "gyro", "rotor_spd", "atti"],  
    )
    # network = Tcn(
    #     input_dim,
    #     output_dim,
    #     [64, 64, 64, 64, 128, 128, 128],
    #     kernel_size=2,
    #     dropout=0.3,
    #     activation="GELU",
    # )
    return network

