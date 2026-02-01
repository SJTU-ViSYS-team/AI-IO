"""
Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/filter/python/src/filter_runner.py
"""

import json

from numba import jit
import numpy as np

from filter.python.src.meas_source_network import MeasSourceNetwork
from filter.python.src.net_input_utils import NetInputBuffer, ImuCalib
from filter.python.src.scekf import ImuMSCKF
from filter.python.src.utils.dotdict import dotdict
from filter.python.src.utils.logging import logging
from filter.python.src.utils.math_utils import mat_exp
from filter.python.src.utils.misc import from_usec_to_sec, from_sec_to_usec


class FilterRunner:
    """
    FilterRunner is responsible for feeding the EKF with the correct data
    It receives the imu measurement, fills the buffer, runs the network with imu data in buffer
    and drives the filter.
    """

    def __init__(
        self,
        model_path,
        model_param_path,
        update_freq,
        filter_tuning,
        imu_calib_dic=None,
        force_cpu=False,
    ):
        config_from_network = dotdict({})
        with open(model_param_path) as json_file:
            data_json = json.load(json_file)
            config_from_network["imu_freq_net"] = data_json["sampling_freq"]
            config_from_network["window_time"] = data_json["window_time"]

        # frequencies and sizes conversion
        self.imu_freq_net = config_from_network.imu_freq_net
        window_size = int(
            (config_from_network.window_time * config_from_network.imu_freq_net) )
        self.net_input_size = window_size

        if not (config_from_network.imu_freq_net / update_freq).is_integer():
            raise ValueError("update_freq must be divisible by imu_freq_net.")
        if not (config_from_network.window_time * update_freq).is_integer():
            raise ValueError("window_time cannot be represented by integer number of updates.")
        self.update_freq = update_freq

        # time
        self.dt_interp_us = int(1.0 / self.imu_freq_net * 1e6)
        self.dt_update_us = int(1.0 / self.update_freq * 1e6)
        self.dt_window_us = int(config_from_network.window_time * 1e6)

        # logging
        logging.info(
            f"Network Input Time: {config_from_network.window_time} (s)"
        )
        logging.info(
            f"Network Input size: {self.net_input_size} (samples)"
        )
        logging.info("IMU and rotor speed input to the network frequency: %s (Hz)" % self.imu_freq_net)
        logging.info("Measurement update frequency: %s (Hz)" % self.update_freq)
        logging.info(
            f"Interpolating IMU and rotor speed measurements every {self.dt_interp_us} [us] for the network input"
        )

        # IMU initial calibration
        self.icalib = ImuCalib()
        self.icalib.from_dic(imu_calib_dic)
        # MSCKF
        self.filter = ImuMSCKF(filter_tuning)

        self.meas_source = MeasSourceNetwork(model_path, force_cpu)

        self.inputs_buffer = NetInputBuffer()

        # This callback is called at first update to initialize the filter
        self.callback_first_update = None

        # keep track of past timestamp and measurement
        self.last_t_us, self.last_acc, self.last_gyr = -1, None, None
        self.last_rotor = None
        self.next_interp_t_us = None
        self.next_update_t_us = None
        self.has_done_first_update = False

    # Note, imu meas for the net are calibrated with offline calibration.
    @jit(forceobj=True, parallel=False, cache=False)
    def _get_inputs_samples_for_network(self, t_begin_us, t_end_us):
        # extract corresponding network input data
        net_ts_begin = t_begin_us
        net_ts_end = t_end_us - self.dt_interp_us

        net_accl, net_gyr, net_rotor, net_t_us = self.inputs_buffer.get_data_from_to(
            net_ts_begin, net_ts_end
        )

        assert net_gyr.shape[0] == self.net_input_size
        assert net_accl.shape[0] == self.net_input_size
        assert net_rotor.shape[0] == self.net_input_size
        
        net_t_s = from_usec_to_sec(net_t_us)

        return net_accl, net_gyr, net_rotor, net_t_s

    def on_imu_measurement(self, t_us, gyr_raw, acc_raw, rotor_spd):
        if self.filter.initialized:
            return self._on_imu_measurement_after_init(t_us, gyr_raw, acc_raw, rotor_spd)
        else:
            logging.info(f"Initializing filter at time {t_us} [us]")
            if self.icalib:
                logging.info(f"Using bias from initial calibration")
                init_ba = self.icalib.accelBias
                init_bg = self.icalib.gyroBias
                # calibrate raw imu data
                acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                    acc_raw, gyr_raw) 
            else:
                logging.info(f"Using zero bias")
                init_ba = np.zeros((3,1))
                init_bg = np.zeros((3,1))
                acc_biascpst, gyr_biascpst = acc_raw, gyr_raw

            self.filter.initialize(acc_biascpst, t_us, init_ba, init_bg)
            self.next_interp_t_us = t_us
            self.next_update_t_us = t_us
            self._add_interpolated_inputs_to_buffer(acc_biascpst, gyr_biascpst, t_us)
            self.next_update_t_us = t_us + self.dt_update_us
            self.last_t_us, self.last_acc, self.last_gyr = (
                t_us,
                acc_biascpst,
                gyr_biascpst,
            )
            self.last_rotor = rotor_spd
            return False

    def _on_imu_measurement_after_init(self, t_us, gyr_raw, acc_raw, rotor_spd):
        """
        For new IMU measurement, after the filter has been initialized
        """
        # Eventually calibrate
        if self.icalib:
            # calibrate raw imu data with offline calibation
            # this is used for network feeding
            acc_biascpst, gyr_biascpst = self.icalib.calibrate_raw(
                acc_raw, gyr_raw
            )

            # calibrate raw imu data with offline calibation scale
            # this is used for the filter. 
            acc_raw, gyr_raw = self.icalib.scale_raw(
                acc_raw, gyr_raw
            )  # only offline scaled - into the filter
        else:
            acc_biascpst = acc_raw
            gyr_biascpst = gyr_raw

        # decide if we need to interpolate imu data or do update
        do_interpolation_of_imu = t_us >= self.next_interp_t_us
        do_update = t_us >= self.next_update_t_us

        # if update the state, check that we compute interpolated measurement also
        assert (
            do_update and do_interpolation_of_imu
        ) or not do_update, (
            "Update and interpolation does not match!"
        )

        # propagation
        # Inputs interpolation and data saving for network
        if do_interpolation_of_imu:
            self._add_interpolated_inputs_to_buffer(acc_biascpst, gyr_biascpst, rotor_spd, t_us)
                
        self.filter.propagate(
            acc_raw, gyr_raw, t_us
        )
        # filter update
        did_update = False
        if do_update:
            did_update = self._process_update(t_us)
            # plan next update of state
            self.next_update_t_us += self.dt_update_us

        # set last value memory to the current one
        self.last_t_us, self.last_acc, self.last_gyr = t_us, acc_biascpst, gyr_biascpst
        self.last_rotor = rotor_spd

        return did_update

    def _process_update(self, t_us):
        t_end_us = t_us
        t_begin_us = t_end_us - self.dt_window_us

        # If we do not have enough IMU data yet, just wait for next time
        if t_begin_us < self.inputs_buffer.net_t_us[0]:
            return False
        # initialize with ground truth at the first update
        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)

        # get measurement from network
        net_accl_b, net_gyr_b, net_rotor, net_t_s = self._get_inputs_samples_for_network(
            t_begin_us, t_end_us)


        meas, meas_cov = self.meas_source.get_measurement(
            net_t_s, net_accl_b, net_gyr_b, net_rotor)
        # filter update
        is_available, innovation, jac, noise_mat = \
            self.filter.learnt_model_update(meas, meas_cov)
        success = False
        if is_available:
            success = self.filter.apply_update(innovation, jac, noise_mat)

        self.has_done_first_update = True
        self.inputs_buffer.throw_data_before(t_begin_us)
        return success

    def _add_interpolated_inputs_to_buffer(self, accl_biascpst, gyr_biascpst, rotor_spd, t_us):

        self.inputs_buffer.add_data_interpolated(
            self.last_t_us,
            t_us,
            self.last_gyr,
            gyr_biascpst,
            self.last_acc,
            accl_biascpst,
            self.last_rotor,
            rotor_spd,
            self.next_interp_t_us,
        )
        self.next_interp_t_us += self.dt_interp_us
