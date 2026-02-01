"""
Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/filter/python/src/net_input_utils.py
"""

import numpy as np
from scipy.interpolate import interp1d


class ImuCalib:
    def __init__(self):
        self.accelScaleInv = np.eye(3)
        self.gyroScaleInv = np.eye(3)
        self.gyroGSense = np.zeros((3,3))
        self.accelBias = np.zeros((3,1))
        self.gyroBias = np.zeros((3,1))

    def from_dic(self, imu_calib_dic):
        self.gyroBias = imu_calib_dic["gyro_bias"].reshape((3,1))
        self.accelBias = imu_calib_dic["accel_bias"].reshape((3,1))

    def calibrate_raw(self, acc, gyr):
        acc_cal = np.dot(self.accelScaleInv, acc) - self.accelBias
        gyr_cal = (
            np.dot(self.gyroScaleInv, gyr)
            - np.dot(self.gyroGSense, acc)
            - self.gyroBias
        )
        return acc_cal, gyr_cal

    def scale_raw(self, acc, gyr):
        acc_cal = np.dot(self.accelScaleInv, acc)
        gyr_cal = np.dot(self.gyroScaleInv, gyr) - np.dot(self.gyroGSense, acc)
        return acc_cal, gyr_cal


class NetInputBuffer:
    """ This is a buffer for interpolated net input data data."""

    def __init__(self):
        self.net_t_us = np.array([])
        self.net_accl = np.array([])
        self.net_gyr = np.array([])
        self.net_rotor = np.array([])

    def add_data_interpolated(
        self, last_t_us, t_us, last_gyr, gyr, last_accl, accl, last_rotor, rotor, requested_interpolated_t_us
    ):
        assert isinstance(last_t_us, int)
        assert isinstance(t_us, int)

        if last_t_us < 0:
            accl_interp = accl.T
            gyr_interp = gyr.T
            rotor_interp = rotor.T
        else:
            try:
                accl_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_accl.T, accl.T]), axis=0)(requested_interpolated_t_us)
                gyr_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_gyr.T, gyr.T]), axis=0)(requested_interpolated_t_us)
                rotor_interp = interp1d(
                    np.array([last_t_us, t_us], dtype=np.uint64).T,
                    np.concatenate([last_rotor.T, rotor.T]), axis=0)(requested_interpolated_t_us)
            except ValueError as e:
                print(
                    f"Trying to do interpolation at {requested_interpolated_t_us} between {last_t_us} and {t_us}"
                )
                raise e
        self._add_data(requested_interpolated_t_us, accl_interp, gyr_interp, rotor_interp)

    def _add_data(self, t_us, accl, gyr, rotor):
        assert isinstance(t_us, int)
        if len(self.net_t_us) > 0:
            assert (
                t_us > self.net_t_us[-1]
            ), f"trying to insert a data at time {t_us} which is before {self.net_t_us[-1]}"

        self.net_t_us = np.append(self.net_t_us, t_us)
        self.net_accl = np.append(self.net_accl, accl).reshape(-1, 3)
        self.net_gyr = np.append(self.net_gyr, gyr).reshape(-1, 3)
        self.net_rotor = np.append(self.net_rotor, rotor).reshape(-1, 4)

    # get network data by input size, extract from the latest
    def get_last_k_data(self, size):
        net_accl = self.net_accl[-size:, :]
        net_gyr = self.net_gyr[-size:, :]
        net_rotor = self.net_rotor[-size:, :]
        net_t_us = self.net_t_us[-size:]
        return net_accl, net_gyr, net_rotor, net_t_us

    # get network data from beginning and end timestamps
    def get_data_from_to(self, t_begin_us: int, t_us_end: int):
        """ This returns all the data from ts_begin to ts_end """
        assert isinstance(t_begin_us, int)
        assert isinstance(t_us_end, int)

        begin_idx = np.argmin(np.abs(self.net_t_us - t_begin_us))
        end_idx   = np.argmin(np.abs(self.net_t_us - t_us_end))

        if abs(self.net_t_us[begin_idx] - t_begin_us) > 1000:
            raise ValueError(f"No suitable begin_idx found within 1ms for t_begin_us={t_begin_us}")
        if abs(self.net_t_us[end_idx] - t_us_end) > 1000:
            raise ValueError(f"No suitable end_idx found within 1ms for t_us_end={t_us_end}")
        net_accl = self.net_accl[begin_idx : end_idx + 1, :]
        net_gyr = self.net_gyr[begin_idx : end_idx + 1, :]
        net_rotor = self.net_rotor[begin_idx : end_idx + 1, :]
        net_t_us = self.net_t_us[begin_idx : end_idx + 1]
        return net_accl, net_gyr, net_rotor, net_t_us

    def throw_data_before(self, t_begin_us: int):
        """ throw away data with timestamp before ts_begin
        """
        assert isinstance(t_begin_us, int)
        begin_idx = np.argmin(np.abs(self.net_t_us - t_begin_us))
        self.net_accl = self.net_accl[begin_idx:, :]
        self.net_gyr = self.net_gyr[begin_idx:, :]
        self.net_rotor = self.net_rotor[begin_idx:, :]
        self.net_t_us = self.net_t_us[begin_idx:]

    def total_net_data(self):
        return self.net_t_us.shape[0]

    def debugstring(self, query_us):
        print(f"min:{self.net_t_us[0]}")
        print(f"max:{self.net_t_us[-1]}")
        print(f"que:{query_us}")
        print(f"all:{self.net_t_us}")

