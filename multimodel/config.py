import torch

class Config:
    def __init__(self):
        self.model_name = "Ablation_AlignOnly"  # 修改名字方便区分
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 12345
        
        self.text_input_dim = 300
        self.vision_input_dim = 35
        self.audio_input_dim = 74
        self.seq_len = 50
        
        self.lstm_layers = 2
        self.hidden_dim = 128
        
        self.emotion_feature_dim = 64
        self.emotion_agnostic_dim = 64
        self.emotion_dim = 64
        
        self.num_emotions = 1
        self.discriminator_hidden = 32
        
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.num_epochs = 60
        self.weight_decay = 1e-4
        self.dropout = 0.4
        
        # --- 实验 1 设置：全部为 0 ---
        self.emotion_loss_weight = 1.0
        self.decouple_loss_weight = 0.01
        self.alignment_loss_weight = 0.01 

config = Config()