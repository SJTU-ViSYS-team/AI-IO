import torch
from learning.network.model_factory import get_model

# ===== 加载模型 =====
weight_path = "/home/csf/LearnedInertialOdometry/results/our2/checkpoints/model_net/checkpoint_with_perturb_rot.pt"
checkpoint = torch.load(weight_path, map_location="cpu")

# 用训练时相同的参数初始化模型
model = get_model(0, 0, 0)  # 这里要改成训练时的参数
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ===== 导出 ONNX =====
dummy_input = torch.randn(1, 16, 100)  # 要用和训练时相同的输入维度
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['vb_learnt', 'vb_cov_learned']
)
