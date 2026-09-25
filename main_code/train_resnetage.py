import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR      = os.environ.get('GA_BASE', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR      = os.path.join(BASE_DIR, 'data/all-organs4/all_organs')
ALTUMAGE_CPGS = os.path.join(BASE_DIR, 'data/multi_platform_cpgs.pkl')
BASELINES_DIR = os.path.join(BASE_DIR, 'baselines')
OUTPUT_PATH   = os.path.join(BASELINES_DIR, 'resnetage_results.csv')
CKPT_PATH     = os.path.join(BASELINES_DIR, 'resnetage_best.pth')

os.makedirs(BASELINES_DIR, exist_ok=True)

SEED         = 0
K_FOLDS      = 5
DESIRED_FOLD = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

BATCH_SIZE    = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 500

print('Loading CpG list...')
AltumAge_cpgs = np.array(pd.read_pickle(ALTUMAGE_CPGS)).tolist()
INPUT_DIM = len(AltumAge_cpgs)
print(f'  {INPUT_DIM} CpG sites')

print('\nLoading methylation data...')

def select(d):
    a = d[d.columns[d.columns.isin(AltumAge_cpgs)].tolist() +
          ['age', 'gender', 'dataset', 'tissue_type']]
    return a[a['tissue_type'].str.lower().str.contains('blood')].dropna()

train_frames, test_frames = [], []
for filename in os.listdir(DATA_DIR):
    if filename.endswith('.pkl'):
        df = select(pd.read_pickle(os.path.join(DATA_DIR, filename)))
        if len(df) <= 0:
            continue
        tr, te = train_test_split(df, test_size=0.2, random_state=42)
        train_frames.append(tr)
        test_frames.append(te)

train_combined = pd.concat(train_frames)
test_combined  = pd.concat(test_frames)
print(f'  Total test samples: {len(test_combined)}')

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_combined)):
    if fold != DESIRED_FOLD:
        continue
    fold_train = train_combined.iloc[train_idx].sample(frac=1, random_state=42)
    fold_val   = train_combined.iloc[val_idx]
    break

X_train = fold_train.drop(columns=['age','gender','dataset','tissue_type']).astype('float32').values
y_train = fold_train['age'].values.astype('float32')
X_val   = fold_val.drop(columns=['age','gender','dataset','tissue_type']).astype('float32').values
y_val   = fold_val['age'].values.astype('float32')
X_test  = test_combined.drop(columns=['age','gender','dataset','tissue_type']).astype('float32').values
y_test  = test_combined['age'].values.astype('float32')

print(f'  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ELU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ELU(),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetAge(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ELU(),
            nn.BatchNorm1d(32),
        )

        self.res_blocks = nn.Sequential(
            ResBlock(32),
            nn.Conv1d(32, 64, kernel_size=1),
            ResBlock(64),
            nn.Conv1d(64, 128, kernel_size=1),
            ResBlock(128),
            nn.Conv1d(128, 128, kernel_size=1),
            ResBlock(128),
            nn.Conv1d(128, 64, kernel_size=1),
            ResBlock(64),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        x = self.head(x)
        return x.squeeze(-1)


train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

model     = ResnetAge(input_dim=INPUT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nModel parameters: {n_params:,}')

best_val_mae = float('inf')
best_epoch   = 0

print(f'Training ResnetAge for {MAX_EPOCHS} epochs...')
for epoch in range(MAX_EPOCHS):
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_preds.extend(model(X_batch.to(device)).cpu().numpy())
            val_true.extend(y_batch.numpy())

    val_mae = mean_absolute_error(val_true, val_preds)
    scheduler.step(val_mae)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_epoch   = epoch + 1
        torch.save(model.state_dict(), CKPT_PATH)

    if (epoch + 1) % 10 == 0:
        print(f'  Epoch {epoch+1:3d}/{MAX_EPOCHS}: '
              f'train_loss={np.mean(train_losses):.3f}, '
              f'val_mae={val_mae:.3f} | '
              f'best_val_mae={best_val_mae:.3f} (epoch {best_epoch})')

print(f'\nTraining complete. Best val MAE={best_val_mae:.3f} at epoch {best_epoch}.')

print('\nEvaluating best checkpoint on test set...')
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.eval()

test_preds, test_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        test_preds.extend(model(X_batch.to(device)).cpu().numpy())
        test_true.extend(y_batch.numpy())

test_preds = np.array(test_preds)
test_true  = np.array(test_true)

mae = mean_absolute_error(test_true, test_preds)
mse = mean_squared_error(test_true, test_preds)
r2  = r2_score(test_true, test_preds)

print(f'  MAE={mae:.3f}, MSE={mse:.3f}, R2={r2:.4f}')

pd.DataFrame({
    'true_age': test_true,
    'predicted_age': test_preds
}, index=test_combined.index).to_csv(
    os.path.join(BASELINES_DIR, 'resnetage_predictions.csv'))

pd.DataFrame([{
    'Model': 'ResnetAge (reimplemented)',
    'MAE': round(mae, 4),
    'MSE': round(mse, 4),
    'R2':  round(r2, 4),
    'Best_epoch': best_epoch
}]).set_index('Model').to_csv(OUTPUT_PATH)

print('\n' + '='*50)
print('FINAL RESULT — ResnetAge (reimplemented)')
print('='*50)
print(f'  MAE : {mae:.3f}')
print(f'  MSE : {mse:.3f}')
print(f'  R2  : {r2:.4f}')
print(f'  Best checkpoint from epoch {best_epoch}/{MAX_EPOCHS}')
print(f'\nFiles saved to: {BASELINES_DIR}/')
