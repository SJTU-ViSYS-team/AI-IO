import os
import subprocess
import re

base_dir = "datasets/DIDO"
max_path_len = 0
max_speed = 0
file_max_path = ""
file_max_speed = ""

# 正则匹配路径长度和最大速度
path_len_re = re.compile(r"path length \(m\)\s+([0-9.]+)")
v_max_re = re.compile(r"v_max \(m/s\)\s+([0-9.]+)")

# 遍历 DIDO 目录下的所有子目录
for traj_dir in os.listdir(base_dir):
    traj_path = os.path.join(base_dir, traj_dir, "stamped_groundtruth_imu.txt")
    if not os.path.isfile(traj_path):
        print(f"Skipped: {traj_path} (file not found)")
        continue

    try:
        # 使用 subprocess 运行 evo_traj
        result = subprocess.run(
            ["evo_traj", "tum", traj_path, "-f"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        output = result.stdout

        # 提取信息
        path_len_match = path_len_re.search(output)
        v_max_match = v_max_re.search(output)

        if path_len_match and v_max_match:
            path_len = float(path_len_match.group(1))
            v_max = float(v_max_match.group(1))

            print(f"{traj_dir} | Path: {path_len:.2f} m | Max speed: {v_max:.2f} m/s")

            if path_len > max_path_len:
                max_path_len = path_len
                file_max_path = traj_dir

            if v_max > max_speed:
                max_speed = v_max
                file_max_speed = traj_dir
        else:
            print(f"Could not extract info from {traj_dir}")

    except Exception as e:
        print(f"Error processing {traj_path}: {e}")

# 最终结果
print("\n===== Summary =====")
print(f"Longest trajectory: {file_max_path} ({max_path_len:.2f} m)")
print(f"Highest max speed: {file_max_speed} ({max_speed:.2f} m/s)")
