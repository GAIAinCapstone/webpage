import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

# ------------------------------
# Dataset 정의
# ------------------------------
class AirQualityDataset(Dataset):
    def __init__(self, features_path, targets_path, seq_len=24):
        self.X = pd.read_csv(features_path).values.astype(np.float32)
        self.y = pd.read_csv(targets_path).values.astype(np.float32)
        self.seq_len = seq_len

        # 정규화
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.X = self.scaler_X.fit_transform(self.X)
        self.y = self.scaler_y.fit_transform(self.y)

        self.samples = len(self.X) - seq_len

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+self.seq_len]),
            torch.tensor(self.y[idx+self.seq_len-1])
        )

# ------------------------------
# Transformer 모델 정의
# ------------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.norm(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1]
        out = self.output_linear(x)
        return out

# ------------------------------
# 모델 학습
# ------------------------------
def train_model():
    features_path = 'data/processed/features.csv'
    targets_path = 'data/processed/targets.csv'
    seq_len = 24
    batch_size = 32
    epochs = 100
    lr = 1e-3

    dataset = AirQualityDataset(features_path, targets_path, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.X.shape[1]
    output_dim = dataset.y.shape[1]

    model = SimpleTransformer(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.HuberLoss()

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
        scheduler.step()
        print(f"📉 Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.6f}")

    os.makedirs("src/models", exist_ok=True)
    torch.save(model.state_dict(), 'src/models/transformer_model.pt')
    print("✅ 모델 학습 및 저장 완료!")

    return model, dataset

# ------------------------------
# 예측 및 시각화
# ------------------------------
def evaluate_model(model, dataset):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i in range(len(dataset)):
            X, y = dataset[i]
            pred = model(X.unsqueeze(0))  # (1, seq_len, input_dim)
            all_preds.append(pred.squeeze(0).numpy())
            all_targets.append(y.numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # 역정규화
    preds_orig = dataset.scaler_y.inverse_transform(preds)
    targets_orig = dataset.scaler_y.inverse_transform(targets)

    # 평가지표
    mae = mean_absolute_error(targets_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
    print(f"✅ MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # 시각화
    pollutant_names = ['NOx', 'SOx', 'TSP']
    for i in range(preds.shape[1]):
        plt.figure(figsize=(12, 4))
        plt.plot(targets_orig[:, i], label='Actual')
        plt.plot(preds_orig[:, i], label='Predicted')
        plt.title(f'{pollutant_names[i]} Prediction')
        plt.xlabel('Time Step')
        plt.ylabel('Concentration')
        plt.legend()
        plt.tight_layout()
        plt.show()

# ------------------------------
# 실행
# ------------------------------
if __name__ == "__main__":
    model, dataset = train_model()
    evaluate_model(model, dataset)
