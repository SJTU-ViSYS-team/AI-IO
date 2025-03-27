# compare data before and after interpolation
import h5py
import numpy as np
ori_data = "datasets/DIDO/random_a_0.7_v_1.5_n_1.5_2022-02-22-13-02-02(0)/data_original.hdf5"
interp_data = "datasets/DIDO/random_a_0.7_v_1.5_n_1.5_2022-02-22-13-02-02(0)/data.hdf5"

ori_acc= []
interp_acc = []

ori_gyro = []
interp_gyro = []

with h5py.File(ori_data, "r") as file:
    print(file.keys())
    ori_acc = np.copy(file["acc"])[:1000]
    ori_gyro = np.copy(file["gyr"])[:1000]

with h5py.File(interp_data, "r") as file:
    interp_acc = np.copy(file["accel_raw"])[:1000]
    interp_gyro = np.copy(file["gyro_raw"])[:1000]

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(ori_acc[:, 0], label="Ori_ax")
plt.plot(interp_acc[:, 0], label="Interp_ax")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(ori_acc[:, 1], label="Ori_ay")
plt.plot(interp_acc[:, 1], label="Interp_ay")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(ori_acc[:, 2], label="Ori_az")
plt.plot(interp_acc[:, 2], label="Interp_az")

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(ori_gyro[:, 0], label="Ori_gx")
plt.plot(interp_gyro[:, 0], label="Interp_gx")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(ori_gyro[:, 1], label="Ori_gy")
plt.plot(interp_gyro[:, 1], label="Interp_gy")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(ori_gyro[:, 2], label="Ori_gz")
plt.plot(interp_gyro[:, 2], label="Interp_gz")
plt.legend()

plt.show()

