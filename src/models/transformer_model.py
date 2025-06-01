import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class AirQualityDataset(Dataset):
    def __init__(self, features_path, targets_path, seq_len=24):
        self.X = pd.read_csv(features_path).values.astype(np.float32)
        self.y = pd.read_csv(targets_path).values.astype(np.float32)
        self.seq_len = seq_len
        self.samples = len(self.X) - seq_len
    def __len__(self):
        return self.samples
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len]),
            torch.tensor(self.y[idx+self.seq_len-1])
        )

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, output_dim)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_linear(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x[-1]  # 마지막 시점만 사용
        out = self.output_linear(x)
        return out

def train_model():
    # 하이퍼파라미터
    features_path = 'data/processed/features.csv'
    targets_path = 'data/processed/targets.csv'
    seq_len = 24
    batch_size = 32
    epochs = 10
    lr = 1e-3

    # 데이터셋
    dataset = AirQualityDataset(features_path, targets_path, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.X.shape[1]
    output_dim = dataset.y.shape[1]
    model = SimpleTransformer(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.6f}")
    torch.save(model.state_dict(), 'src/models/transformer_model.pt')
    print("모델 학습 및 저장 완료!")

if __name__ == "__main__":
    train_model() 