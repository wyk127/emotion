import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pickle
import os
import sys
import logging
import random
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

from config import config
from model import MultimodalEmotionAnalyzer

# --- 0. 辅助工具 ---
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- 1. 数据集类 ---
class MOSIDataset(Dataset):
    def __init__(self, pkl_path, split='train', mean_std=None):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"找不到文件: {pkl_path}")
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.vision = np.nan_to_num(data[split]['vision'], nan=0.0)
        self.text = np.nan_to_num(data[split]['text'], nan=0.0)
        self.audio = np.nan_to_num(data[split]['audio'], nan=0.0)
        self.labels = np.nan_to_num(data[split]['labels'], nan=0.0)
        
        if split == 'train':
            self.mean_std = {
                'vision': (np.mean(self.vision), np.std(self.vision) + 1e-6),
                'text': (np.mean(self.text), np.std(self.text) + 1e-6),
                'audio': (np.mean(self.audio), np.std(self.audio) + 1e-6)
            }
        else:
            self.mean_std = mean_std

        if self.mean_std is not None:
            self.vision = (self.vision - self.mean_std['vision'][0]) / self.mean_std['vision'][1]
            self.text = (self.text - self.mean_std['text'][0]) / self.mean_std['text'][1]
            self.audio = (self.audio - self.mean_std['audio'][0]) / self.mean_std['audio'][1]

        self.length = self.labels.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'text': torch.tensor(self.text[idx], dtype=torch.float32),
            'vision': torch.tensor(self.vision[idx], dtype=torch.float32),
            'audio': torch.tensor(self.audio[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# --- 2. 评估函数 (增加 Debug 输出) ---
def evaluate(model, data_loader, device, show_sample=False):
    model.eval()
    total_loss = 0
    preds = []
    truths = []
    
    with torch.no_grad():
        for batch in data_loader:
            t = batch['text'].to(device)
            v = batch['vision'].to(device)
            a = batch['audio'].to(device)
            y = batch['label'].to(device).view(-1, 1)
            
            outputs = model(t, v, a)
            pred = outputs['pred']
            total_loss += torch.abs(pred - y).sum().item()
            preds.append(pred.cpu().numpy())
            truths.append(y.cpu().numpy())
    
    if len(preds) == 0: return 0, 0, 0
    
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    avg_loss = total_loss / len(data_loader.dataset)
    
    # 打印前5个样本看看模型在预测什么
    if show_sample:
        print("\n--- Sample Predictions ---")
        for i in range(min(5, len(preds))):
            print(f"Pred: {preds[i][0]:.4f} | Truth: {truths[i][0]:.4f}")
    
    preds_binary = (preds >= 0).astype(int)
    truths_binary = (truths >= 0).astype(int)
    
    acc = accuracy_score(truths_binary, preds_binary)
    f1 = f1_score(truths_binary, preds_binary, average='weighted')
    
    return avg_loss, acc, f1

# --- 3. 主训练循环 ---
def train():
    pkl_path = r"F:\code\multimodel\mosi_data_fixed.pkl" 
    save_dir = "checkpoints"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(os.path.join(save_dir, f"train_{timestamp}.log"))
    setup_seed(config.seed)
    
    logger.info("Loading datasets...")
    train_dataset = MOSIDataset(pkl_path, 'train')
    valid_dataset = MOSIDataset(pkl_path, 'valid', mean_std=train_dataset.mean_std)
    test_dataset = MOSIDataset(pkl_path, 'test', mean_std=train_dataset.mean_std)

    # --- 诊断信息 ---
    logger.info(f"Label Stats | Mean: {np.mean(train_dataset.labels):.4f} | Std: {np.std(train_dataset.labels):.4f}")
    logger.info(f"Label Range | Min: {np.min(train_dataset.labels):.4f} | Max: {np.max(train_dataset.labels):.4f}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = MultimodalEmotionAnalyzer().to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(save_dir, 'best_model.pth'), trace_func=logger.info)

    logger.info("Start Training...")
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss_sum = 0
        steps = 0
        
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{config.num_epochs}]")
        
        for batch in loop:
            t = batch['text'].to(config.device)
            v = batch['vision'].to(config.device)
            a = batch['audio'].to(config.device)
            y = batch['label'].to(config.device)
            
            optimizer.zero_grad()
            outputs = model(t, v, a)
            
            loss, task_loss = model.get_loss(outputs, y)
            if torch.isnan(loss): continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_sum += task_loss.item()
            steps += 1
            loop.set_postfix(loss=loss.item(), task_loss=task_loss.item())
            
        avg_train_loss = train_loss_sum / steps if steps > 0 else 0
        
        # 每个 Epoch 打印一次样本，看看模型有没有在学
        show_sample = (epoch % 5 == 0) 
        val_mae, val_acc, val_f1 = evaluate(model, valid_loader, config.device, show_sample=show_sample)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f} | Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_mae)
        early_stopping(val_mae, model)
        
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break
            
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    test_mae, test_acc, test_f1 = evaluate(model, test_loader, config.device, show_sample=True)
    logger.info(f"FINAL TEST RESULTS | MAE: {test_mae:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

if __name__ == "__main__":
    train()