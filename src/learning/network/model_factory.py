from learning.network.model import Tcn, IMUTransformerWithModality


def get_model(window_s=100):
    network = IMUTransformerWithModality(
        sub_dim=16,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        output_size=3,
        window_size=window_s,
        enabled_modalities=["acc", "gyro", "rotor_spd"]
    )
    # network = Tcn(
    #     input_size=10,
    #     output_size=3,
    #     num_channels=[64, 64, 128],
    #     kernel_size=2,
    #     dropout=0.3,
    #     activation="GELU",
    # )
    return network

