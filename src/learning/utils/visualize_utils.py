import matplotlib.pyplot as plt
import os

import torch
import seaborn as sns

def plot_modality_activation(activations, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    for name, act in activations.items():
        mean_act = act.mean(dim=(0, 1))  # shape: (sub_dim,)
        plt.figure(figsize=(6, 3))
        plt.bar(range(len(mean_act)), mean_act)
        plt.title(f"[{name}] Feature Activation")
        plt.xlabel("Feature Index")
        plt.ylabel("Mean Activation")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}_activation_epoch{epoch}.png"))
        plt.close()

def plot_attention_heatmap(attn_weights, save_dir, epoch, layer_idx=0, head_idx=0):
    os.makedirs(save_dir, exist_ok=True)
    attn = attn_weights[layer_idx][0, head_idx].cpu()  # (T, T)
    plt.figure(figsize=(6, 5))
    plt.imshow(attn, cmap='viridis')
    plt.title(f"Attention Heatmap - Layer {layer_idx} Head {head_idx}")
    plt.xlabel("Key Time Step")
    plt.ylabel("Query Time Step")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"attn_epoch{epoch}_layer{layer_idx}_head{head_idx}.png"))
    plt.close()

def plot_feature_trend_over_time(feature_tensor, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    feature_tensor = feature_tensor.cpu().numpy()
    plt.figure(figsize=(10, 4))
    for d in range(min(10, feature_tensor.shape[1])):  # plot first 10 dims
        plt.plot(feature_tensor[:, d], label=f'Dim {d}')
    plt.title("Fused Feature Timeline")
    plt.xlabel("Time Step")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"feature_trend_epoch{epoch}.png"))
    plt.close()

def plot_attention_to_last_frame(attn_weights, save_dir, epoch, layer_idx=0, head_idx=0):
    os.makedirs(save_dir, exist_ok=True)

    attn = attn_weights[layer_idx][0, head_idx].cpu()  # (T, T)
    last_frame_attention = attn[:, -1]  # every query attention to key=T-1

    plt.figure(figsize=(6, 3))
    plt.plot(last_frame_attention.numpy())
    plt.title(f"Attention to Last Frame - Layer {layer_idx} Head {head_idx}")
    plt.xlabel("Query Time Step")
    plt.ylabel("Attention to Last Frame")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"attn_to_last_epoch{epoch}.png"))
    plt.close()

def plot_modality_contribution_heatmap(activations, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)

    # mean activation across (B, T, D)
    contrib_dict = {name: act.abs().mean(dim=(0, 1)).cpu().numpy() for name, act in activations.items()}  
    contrib_tensor = torch.stack([torch.tensor(v) for v in contrib_dict.values()], dim=0)  # (num_modalities, dim)

    modality_names = list(contrib_dict.keys())

    plt.figure(figsize=(10, 4))
    sns.heatmap(contrib_tensor, xticklabels=False, yticklabels=modality_names, cmap='viridis', annot=False)
    plt.title("Modality Contribution Heatmap")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Modality")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"modality_contribution_epoch{epoch}.png"))
    plt.close()

def plot_modality_time_heatmap(activations, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    modality_names = list(activations.keys())
    B, T, _ = list(activations.values())[0].shape

    modality_strengths = []
    for name in modality_names:
        act = activations[name]  # (B, T, D)
        strength = act.abs().mean(dim=(0, 2))  # (T,)
        modality_strengths.append(strength)

    modality_strengths = torch.stack(modality_strengths, dim=0).cpu().numpy()  # (M, T)

    plt.figure(figsize=(10, 4))
    plt.imshow(modality_strengths, cmap='plasma', aspect='auto')
    plt.colorbar(label='Mean Activation')
    plt.yticks(range(len(modality_names)), modality_names)
    plt.xlabel("Time Step")
    plt.ylabel("Modality")
    plt.title("Modality Contribution over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"modality_time_heatmap_epoch{epoch}.png"))
    plt.close()

def visualize_net(network, save_dir, epoch):
    # visualize activation of different modalities
    plot_modality_activation(network.activations, save_dir, epoch)

    plot_modality_contribution_heatmap(network.activations, save_dir, epoch)
    
    plot_modality_time_heatmap(network.activations, save_dir, epoch)

    # visualize attention weights
    attn_weights = [layer.attn_weights for layer in network.transformer.layers]
    plot_attention_heatmap(attn_weights, save_dir, epoch, layer_idx=0, head_idx=0)

    plot_attention_to_last_frame(attn_weights, save_dir, epoch)

    # visualize feature trend over time
    fused_feature = torch.cat([v for v in network.activations.values()], dim=-1)  # (B, T, D)
    plot_feature_trend_over_time(fused_feature[0], save_dir, epoch)