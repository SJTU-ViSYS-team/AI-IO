import rosbag
import rospy
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def get_time(header):
    return header.stamp.secs + header.stamp.nsecs * 1e-9

def process_bag(bag_path, args):
    print(f"\nProcessing: {bag_path}")
    out_dir = os.path.dirname(bag_path)
    assert os.path.exists(out_dir)

    gra = 9.8
    bag_data = rosbag.Bag(bag_path, "r")

    t0, t1 = None, None
    timestamp_status = []
    thr2acc = []
    for topic, msg, t in bag_data.read_messages("/bfctrl/statue"):
        t = get_time(msg.header)
        if msg is not None:
            if t0 is None and msg.status == 4:
                t0 = t
            if t0 is not None and msg.status == 4:
                timestamp_status.append(t - t0)
                thr2acc.append(gra / msg.hover_percentage)
            if t0 is not None and msg.status != 4 and t1 is None:
                t1 = t
                break

    if t0 is None or t1 is None:
        print("Could not determine valid t0 and t1 — skipping")
        bag_data.close()
        return

    print(f"On BFCTRL_STATUS_CMD mode from {t0} to {t1}, lasting {t1 - t0:.2f} seconds")

    timestamp = []
    ctbr = []
    for topic, msg, t in bag_data.read_messages("/mavros/setpoint_raw/attitude"):
        t = get_time(msg.header)
        if msg is not None and t0 <= t < t1:
            timestamp.append(t - t0)
            ctbr.append([msg.body_rate.x, msg.body_rate.y, msg.body_rate.z, msg.thrust])

    timestamp_vicon, position, orientation, lin_vel, ang_vel = [], [], [], [], []
    for topic, msg, t in bag_data.read_messages("/vicon/yf_imu/odom"):
        t = get_time(msg.header)
        if msg is not None and t0 <= t < t1:
            timestamp_vicon.append(t - t0)
            position.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
            orientation.append([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
            ang_vel.append([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
            lin_vel.append([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])

    timestamp_imu, acc_data, gyr_data = [], [], []
    for topic, msg, t in bag_data.read_messages("/mavros/imu/data"):
        t = get_time(msg.header)
        if msg is not None and t0 <= t < t1:
            timestamp_imu.append(t - t0)
            acc_data.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            gyr_data.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

    bag_data.close()

    print(f"Length of thrust2acc: {len(thr2acc)}, ctbr: {len(ctbr)}, pos: {len(position)}, acc: {len(acc_data)}")

    # === Plotting code omitted for brevity ===
    # Keep your plotting section here if args.show_plot or args.save_plot

    if args.make_dataset:
        name = os.path.basename(bag_path).replace(".bag", ".csv")

        control_data = {
            "t_status(ns)": (np.array(timestamp_status)*1e9).astype(np.int64).tolist(),
            "thrust2acc": thr2acc,
            "t_ctbr": timestamp,
            "cmd_wx": np.array(ctbr)[:, 0].tolist(),
            "cmd_wy": np.array(ctbr)[:, 1].tolist(),
            "cmd_wz": np.array(ctbr)[:, 2].tolist(),
            "cmd_thrust": np.array(ctbr)[:, 3].tolist(),
        }
        max_length = max(len(col) for col in control_data.values())
        for key in control_data:
            control_data[key] += [np.nan] * (max_length - len(control_data[key]))
        df = pd.DataFrame(control_data)
        df.to_csv(os.path.join(out_dir, "control_data.csv"), index=False, float_format="%.6f")

        gt_data = {
            "t_vicon(ns)": (np.array(timestamp_vicon)*1e9).astype(np.int64).tolist(),
            "x": np.array(position)[:, 0].tolist(),
            "y": np.array(position)[:, 1].tolist(),
            "z": np.array(position)[:, 2].tolist(),
            "qw": np.array(orientation)[:, 3].tolist(),
            "qx": np.array(orientation)[:, 0].tolist(),
            "qy": np.array(orientation)[:, 1].tolist(),
            "qz": np.array(orientation)[:, 2].tolist(),
            "vx": np.array(lin_vel)[:, 0].tolist(),
            "vy": np.array(lin_vel)[:, 1].tolist(),
            "vz": np.array(lin_vel)[:, 2].tolist()
        }
        df = pd.DataFrame(gt_data)
        df.to_csv(os.path.join(out_dir, "pose_data.csv"), index=False, float_format="%.6f")

        imu = {
            "t_imu(ns)": (np.array(timestamp_imu)*1e9).astype(np.int64).tolist(),
            "wx": np.array(gyr_data)[:, 0].tolist(),
            "wy": np.array(gyr_data)[:, 1].tolist(),
            "wz": np.array(gyr_data)[:, 2].tolist(),
            "ax": np.array(acc_data)[:, 0].tolist(),
            "ay": np.array(acc_data)[:, 1].tolist(),
            "az": np.array(acc_data)[:, 2].tolist()
        }
        df = pd.DataFrame(imu)
        df.to_csv(os.path.join(out_dir, "imu_data.csv"), index=False, float_format="%.6f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process ROS bag files")
    parser.add_argument("--root", type=str, default="ours", help="Root directory containing subdirectories with .bag files")
    parser.add_argument("--show_plot", action="store_true", default=False, help="Plot the contents of the bag file")
    parser.add_argument("--save_plot", action="store_true", default=False, help="Save the figure")
    parser.add_argument("--make_dataset", action="store_true", default=True, help="Make dataset from the bag file")
    args = parser.parse_args()

    for subdir, _, files in os.walk(args.root):
        for file in files:
            if file.endswith(".bag"):
                bag_path = os.path.join(subdir, file)
                try:
                    process_bag(bag_path, args)
                except Exception as e:
                    print(f"Error processing {bag_path}: {e}")
