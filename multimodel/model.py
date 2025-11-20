import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import config

# --- 1. 黄金时序编码器 (LSTM + Attention) ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, gru_output):
        attention_scores = self.attention_layer(gru_output)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * gru_output, dim=1)
        return context_vector, attention_weights

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(TemporalEncoder, self).__init__()
        # 移除了 Input LayerNorm，这对 MOSI 很重要
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = Attention(hidden_dim * 2)
        self.project = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        context, _ = self.attention(out)
        projected = self.project(context)
        return self.layer_norm(projected)

# --- 2. 解耦器 (GELU 版本) ---
class FeatureDecoupler(nn.Module):
    def __init__(self, input_dim, emotion_dim, agnostic_dim):
        super(FeatureDecoupler, self).__init__()
        self.emotion_projector = nn.Sequential(
            nn.Linear(input_dim, emotion_dim * 2),
            nn.LayerNorm(emotion_dim * 2),
            nn.GELU(),
            nn.Linear(emotion_dim * 2, emotion_dim)
        )
        self.agnostic_projector = nn.Sequential(
            nn.Linear(input_dim, agnostic_dim * 2),
            nn.LayerNorm(agnostic_dim * 2),
            nn.GELU(),
            nn.Linear(agnostic_dim * 2, agnostic_dim)
        )
        self.reconstructor = nn.Linear(emotion_dim + agnostic_dim, input_dim)
        
    def forward(self, x):
        emotion_feat = self.emotion_projector(x)
        agnostic_feat = self.agnostic_projector(x)
        combined_feat = torch.cat([emotion_feat, agnostic_feat], dim=-1)
        reconstructed = self.reconstructor(combined_feat)
        return emotion_feat, agnostic_feat, reconstructed

# --- 3. 判别器 & 对齐器 ---
class EmotionDiscriminator(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(EmotionDiscriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1) 
        )
    def forward(self, x):
        return self.classifier(x)

class CrossModalAligner(nn.Module):
    def __init__(self, dim):
        super(CrossModalAligner, self).__init__()
        self.v_proj = nn.Linear(dim, dim)
        self.t_proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, v, t):
        # 前向传播不归一化，保留强度
        v_p = self.v_proj(v)
        t_p = self.t_proj(t)
        # 计算 Loss 时才归一化
        v_norm = F.normalize(v_p, dim=-1)
        t_norm = F.normalize(t_p, dim=-1)
        sim = torch.matmul(v_norm, t_norm.T) * self.temperature.exp()
        return v_p, t_p, sim

# --- 4. 主模型 (回归简单融合) ---
class MultimodalEmotionAnalyzer(nn.Module):
    def __init__(self):
        super(MultimodalEmotionAnalyzer, self).__init__()
        
        self.dim = 128 
        
        self.text_encoder = TemporalEncoder(config.text_input_dim, self.dim, num_layers=config.lstm_layers)
        self.vision_encoder = TemporalEncoder(config.vision_input_dim, self.dim, num_layers=config.lstm_layers)
        self.audio_encoder = TemporalEncoder(config.audio_input_dim, self.dim, num_layers=config.lstm_layers)

        self.vision_decoupler = FeatureDecoupler(self.dim, 64, 64)
        self.text_decoupler = FeatureDecoupler(self.dim, 64, 64)
        
        self.vision_discriminator = EmotionDiscriminator(64, 32)
        self.text_discriminator = EmotionDiscriminator(64, 32)
        self.aligner = CrossModalAligner(64)
        
        self.audio_proj = nn.Linear(self.dim, 64)
        
        # 简单的 Concat + MLP 融合
        # 输入: T_aligned(64) + V_aligned(64) + A_proj(64) = 192
        fusion_dim = 64 * 3
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.dropout), # 这里的 Dropout 很关键
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, text_seq, vision_seq, audio_seq):
        t = self.text_encoder(text_seq)
        v = self.vision_encoder(vision_seq)
        a = self.audio_encoder(audio_seq)
        
        v_emo, v_agn, v_rec = self.vision_decoupler(v)
        t_emo, t_agn, t_rec = self.text_decoupler(t)
        
        v_aligned, t_aligned, sim_matrix = self.aligner(v_emo, t_emo)
        
        a_proj = self.audio_proj(a)
        
        # 简单粗暴的拼接，往往最有效
        feat_to_fuse = torch.cat([t_aligned, v_aligned, a_proj], dim=-1)
        prediction = self.fusion_layer(feat_to_fuse)
        
        v_disc = self.vision_discriminator(v_agn)
        t_disc = self.text_discriminator(t_agn)
        
        return {
            'pred': prediction,
            'v_embed': v, 't_embed': t,
            'v_rec': v_rec, 't_rec': t_rec,
            'v_emo': v_emo, 't_emo': t_emo,
            'v_agn': v_agn, 't_agn': t_agn,
            'v_disc': v_disc, 't_disc': t_disc,
            'sim_matrix': sim_matrix
        }

    def get_loss(self, outputs, labels):
        pred = outputs['pred']
        labels = labels.view(-1, 1)
        
        task_loss = F.l1_loss(pred, labels)
        
        recon_loss = F.mse_loss(outputs['v_rec'], outputs['v_embed']) + \
                     F.mse_loss(outputs['t_rec'], outputs['t_embed'])
        
        disc_loss = F.mse_loss(outputs['v_disc'], labels) + \
                    F.mse_loss(outputs['t_disc'], labels)
        adv_loss = -disc_loss 
        
        orth_loss = torch.mean(torch.sum(outputs['v_emo'] * outputs['v_agn'], dim=1)**2) + \
                    torch.mean(torch.sum(outputs['t_emo'] * outputs['t_agn'], dim=1)**2)
        
        sim_matrix = outputs['sim_matrix']
        batch_size = sim_matrix.size(0)
        targets = torch.arange(batch_size).to(sim_matrix.device)
        align_loss = (F.cross_entropy(sim_matrix, targets) + F.cross_entropy(sim_matrix.T, targets)) / 2
        
        # 辅助权重保持微小，不干扰主任务
        total_loss = config.emotion_loss_weight * task_loss + \
                     config.decouple_loss_weight * (recon_loss + 0.001 * adv_loss + orth_loss) + \
                     config.alignment_loss_weight * align_loss
                     
        return total_loss, task_loss