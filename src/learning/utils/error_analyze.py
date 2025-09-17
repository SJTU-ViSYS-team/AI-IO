
import numpy as np


def compute_rmse(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute RMSE between estimated and ground truth data.
    est, gt: arrays of shape (N, D) where D is typically 3 (x, y, z)
    Returns: RMSE for each dimension as a (D + 1,) array
    """
    error = est - gt
    # per-axis RMSE
    mse = np.mean(error ** 2, axis=0)
    rmse_axes = np.sqrt(mse)

    # sum RMSE
    error_sum = np.linalg.norm(error, axis=1)

    rmse = np.sqrt(np.mean(error_sum ** 2))

    return np.concatenate([rmse_axes, [rmse]])

def angular_error_deg(pred, gt):
    """
    pred, gt: [N, 3] in degrees
    returns RMSE with angle wrapping in degrees
    """
    error = pred - gt
    # Wrap to [-180, 180]
    error = (error + 180) % 360 - 180
    rmse = np.sqrt(np.mean(error**2, axis=0))
    return rmse

def compute_position_velocity_orientation_errors(est_pos, gt_pos, est_vel, gt_vel, est_euler, gt_euler):
    """
    Compute RMSE errors for position, velocity, and orientation.
    Returns a dictionary with RMSE values.
    """
    errors = {
        "position_rmse": compute_rmse(est_pos[:, 1:], gt_pos[:, 1:]),
        "velocity_rmse": compute_rmse(est_vel[:, 1:], gt_vel[:, 1:]),
        "orientation_rmse": angular_error_deg(est_euler[:, 1:], gt_euler[:, 1:]), 
        "position_rrmse": compute_rmse(est_pos[500-1:, 1:] - est_pos[:-500+1, 1:], gt_pos[500-1:, 1:] - gt_pos[:-500+1, 1:]),
        "velocity_rrmse": compute_rmse(est_vel[500-1:, 1:] - est_vel[:-500+1, 1:], gt_vel[500-1:, 1:] - gt_vel[:-500+1, 1:]),
        "orientation_rrmse": angular_error_deg(est_euler[500-1:, 1:] - est_euler[:-500+1, 1:], gt_euler[500-1:, 1:] - gt_euler[:-500+1, 1:]) # in degrees

    }
    return errors

def compute_position_velocity_errors(est_pos, gt_pos, est_vel, gt_vel):
    """
    Compute RMSE errors for position, velocity, and orientation.
    Returns a dictionary with RMSE values.
    """
    errors = {
        "position_rmse": compute_rmse(est_pos[:, 1:], gt_pos[:, 1:]),
        "velocity_rmse": compute_rmse(est_vel[:, 1:], gt_vel[:, 1:]),
        "position_rrmse": compute_rmse(est_pos[500-1:, 1:] - est_pos[:-500+1, 1:], gt_pos[500-1:, 1:] - gt_pos[:-500+1, 1:]),
        "velocity_rrmse": compute_rmse(est_vel[500-1:, 1:] - est_vel[:-500+1, 1:], gt_vel[500-1:, 1:] - gt_vel[:-500+1, 1:])

    }
    return errors

def print_rmse_summary(errors):
    """
    Pretty-print the RMSE results.
    """
    print("RMSE Summary:")
    print("-------------")
    print(f"Position RMSE (x, y, z) [m]:     {errors['position_rmse']}")
    print(f"Velocity RMSE (x, y, z) [m/s]:   {errors['velocity_rmse']}")
    print(f"Orientation RMSE (yaw, pitch, roll) [deg]: {errors['orientation_rmse']}")
    print(f"Relative Position RMSE (x, y, z) [m]:     {errors['position_rrmse']}")
    print(f"Relative Velocity RMSE (x, y, z) [m/s]:   {errors['velocity_rrmse']}")
    print(f"Relative Orientation RMSE (yaw, pitch, roll) [deg]: {errors['orientation_rrmse']}")
