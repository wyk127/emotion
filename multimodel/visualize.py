import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from model import MultimodalEmotionAnalyzer
from config import config
from train import MOSIDataset
import os

plt.style.use('seaborn-v0_8-whitegrid')

def plot_embedding(data, label, title, save_name):
    # 归一化
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 8))
    
    colors = ['#FF3333' if l > 0 else '#3333FF' for l in label]
    
    # alpha=0.6, s=30 让点散布更有质感
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=30, alpha=0.6, edgecolors='white', linewidth=0.2)
    
    plt.scatter([], [], c='#FF3333', s=50, label='Positive')
    plt.scatter([], [], c='#3333FF', s=50, label='Negative')
    plt.legend(fontsize=14, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_name}")
    plt.close()

def main():
    print("=== Generating Optimized Visualization ===")
    
    config.decouple_loss_weight = 0.01
    config.alignment_loss_weight = 0.01
    
    pkl_path = r"F:\code\multimodel\mosi_data_fixed.pkl"
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found")
        return
    
    train_dataset = MOSIDataset(pkl_path, 'train') 
    test_dataset = MOSIDataset(pkl_path, 'test', mean_std=train_dataset.mean_std)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = MultimodalEmotionAnalyzer().to(config.device)
    model_path = os.path.join("checkpoints", "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_emo_feat = []
    all_agn_feat = []
    all_labels = []

    print("Extracting features...")
    with torch.no_grad():
        for batch in test_loader:
            t = batch['text'].to(config.device)
            v = batch['vision'].to(config.device)
            a = batch['audio'].to(config.device)
            y = batch['label'].to(config.device)

            outputs = model(t, v, a)
            all_emo_feat.append(outputs['v_emo'].cpu().numpy())
            all_agn_feat.append(outputs['v_agn'].cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_emo_feat = np.concatenate(all_emo_feat, axis=0)
    all_agn_feat = np.concatenate(all_agn_feat, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).squeeze()

    print("Running t-SNE with Cosine Metric (Better for high-dim vectors)...")
    
    # --- 关键修改 ---
    # 1. metric='cosine': 余弦距离更适合这种特征向量，能消除模长的干扰
    # 2. perplexity=5: 降低困惑度，只看非常局部的结构。这会打破那个大C形，让相似的点抱团。
    tsne = TSNE(n_components=2, perplexity=5, metric='cosine', init='random', random_state=42)
    
    print("  - Emotion Features...")
    emo_tsne = tsne.fit_transform(all_emo_feat)
    plot_embedding(emo_tsne, all_labels, 'Visualization of Emotion Features', 'tsne_emotion_opt.png')
    
    print("  - Agnostic Features...")
    # 对 Agnostic 再跑一次 t-SNE，看看能不能散开
    agn_tsne = tsne.fit_transform(all_agn_feat)
    plot_embedding(agn_tsne, all_labels, 'Visualization of Agnostic Features', 'tsne_agnostic_opt.png')
    
    print("Done! Please check the new images.")

if __name__ == "__main__":
    main()