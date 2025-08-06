
import numpy as np


def compute_rmse(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Compute RMSE between estimated and ground truth data.
    est, gt: arrays of shape (N, D) where D is typically 3 (x, y, z)
    Returns: RMSE for each dimension as a (D,) array
    """
    mse = np.mean((est - gt) ** 2, axis=0)
    return np.sqrt(mse)

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
        "orientation_rmse": angular_error_deg(est_euler[:, 1:], gt_euler[:, 1:])  # in degrees
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
