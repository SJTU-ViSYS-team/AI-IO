# 读取原始文件
with open('/home/csf/learned_inertial_model_odometry/datasets/Blackbird/clover/yawForward/maxSpeed5p0/test/stamped_groundtruth_imu.txt', 'r') as file:
    lines = file.readlines()

# 处理每一行，去掉最后三列
processed_lines = []
for line in lines:
    if line.startswith('#'):
        processed_lines.append(line.strip())  # 保留表头
    else:
        columns = line.strip().split()
        processed_line = ' '.join(columns[:-3])  # 去掉最后三列
        processed_lines.append(processed_line)

# 将处理后的数据写入新文件
with open('/home/csf/learned_inertial_model_odometry/datasets/Blackbird/clover/yawForward/maxSpeed5p0/test/gt_imu_tum.txt', 'w') as file:
    for line in processed_lines:
        file.write(line + '\n')

print("处理完成，结果已保存到 output.txt")
