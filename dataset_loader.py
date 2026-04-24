import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader

class RSRPDataset(Dataset):
    def __init__(self, data_dir='data/', seq_len=50, pred_len=10):
        # This will load and combine all three CSV files
        csv_files = [
            'drive_test_measurements01.csv',
            'drive_test_measurements02.csv',
            'drive_test_measurements03.csv'
        ]
        
        data_list = []
        for f in csv_files:
            path = os.path.join(data_dir, f)
            df = pd.read_csv(path)
            # Use 'RSRP' (uppercase) to match the actual column name in the files
            data_list.append(df['RSRP'].values) 
            
        self.data = np.concatenate(data_list).astype(np.float32)
        
        # Normalize to [-1, 1] for Diffusion
        self.min_val = np.min(self.data)
        self.max_val = np.max(self.data)
        self.data = (self.data - self.min_val) / (self.max_val - self.min_val) * 2 - 1
        
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)
