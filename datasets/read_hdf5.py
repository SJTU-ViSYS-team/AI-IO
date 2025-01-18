import h5py

def print_hdf5_structure(name, obj):
    """递归打印每个组和数据集的信息。"""
    # 打印组或数据集的路径
    print(f"\n{name}")
    
    # 如果是组，打印包含的数据集数量
    if isinstance(obj, h5py.Group):
        print("  Group:")
        print("  Contains datasets:", list(obj.keys()))
    # 如果是数据集，打印其形状和数据
    elif isinstance(obj, h5py.Dataset):
        print("  Dataset:")
        print("  Shape:", obj.shape)
        print("  Data type:", obj.dtype)
        
        # 打印数据集内容
        data = obj[:]
        print("  Data:", data)
        
        # 如果数据量过大，只显示前10个元素
        if data.size > 10:
            print("  Partial data (first 10 items):", data.flat[:10])

# 打开 HDF5 文件
with h5py.File("/home/csf/learned_inertial_model_odometry/datasets/Euroc/mav0/processed_data/train/data.hdf5", "r") as file:
    # 遍历并打印整个文件的结构和数据
    file.visititems(print_hdf5_structure)
