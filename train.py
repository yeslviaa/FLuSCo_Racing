import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split


DYNAMIC_FEATURES = ['angle', 'speedX', 'speedY', 'speedZ',
                    'trackPos', 'rpm', 'gear']
TRACK_FEATURES   = [f'track_{i}' for i in range(19)]
INPUT_FEATURES   = DYNAMIC_FEATURES + TRACK_FEATURES        
TARGET_FEATURES  = ['steer', 'accel', 'brake']

INPUT_DIM  = len(INPUT_FEATURES)    
OUTPUT_DIM = len(TARGET_FEATURES)   


BATCH_SIZE     = 256
LEARNING_RATE  = 1e-4
MAX_EPOCHS     = 100
EARLY_STOP_PAT = 15        
VAL_SPLIT      = 0.15      

STEER_WEIGHT = 5.0
PEDAL_WEIGHT = 1.0
LOSS_WEIGHTS = [STEER_WEIGHT, PEDAL_WEIGHT, PEDAL_WEIGHT]  


class DrivingMLP(nn.Module):
 
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.steer_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.pedal_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        features = self.backbone(x)             
        steer    = self.steer_head(features)     
        pedals   = self.pedal_head(features)     
        return torch.cat([steer, pedals], dim=1)


class WeightedMSELoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32).view(1, -1)
        self.register_buffer('weights', w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        squared_error = (pred - target) ** 2          
        weighted      = squared_error * self.weights  
        return weighted.mean()


def load_and_preprocess(csv_path: str, scaler_path: str):
 
    print(f"[DATA] Lettura CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    n0 = len(df)

    missing_in  = [c for c in INPUT_FEATURES  if c not in df.columns]
    missing_out = [c for c in TARGET_FEATURES if c not in df.columns]
    if missing_in or missing_out:
        raise ValueError(
            f"Colonne mancanti nel CSV.\n  Input mancanti:  {missing_in}\n"
            f"  Target mancanti: {missing_out}"
        )

    df = df.dropna(subset=INPUT_FEATURES + TARGET_FEATURES).reset_index(drop=True)
    n1 = len(df)
    print(f"[DATA] Righe lette: {n0}  |  dopo dropna: {n1}  "
          f"(rimosse {n0 - n1})")

    if n1 == 0:
        raise ValueError("Il dataset e' vuoto dopo la rimozione dei NaN.")

    X = df[INPUT_FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET_FEATURES].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    joblib.dump(scaler, scaler_path)
    print(f"[DATA] Scaler salvato in: {scaler_path}")

    return X_scaled, y


def build_dataloaders(X: np.ndarray, y: np.ndarray, seed: int, device: torch.device):
  
    dataset  = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    val_size = int(round(len(dataset) * VAL_SPLIT))
    trn_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [trn_size, val_size], generator=generator)

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=pin)

    print(f"[DATA] Split -> Train: {trn_size}  |  Val: {val_size}")
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss, total_samples = 0.0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        bs = X_batch.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total_loss, total_samples = 0.0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        preds = model(X_batch)
        loss  = criterion(preds, y_batch)

        bs = X_batch.size(0)
        total_loss    += loss.item() * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning Trainer (Multi-Head MLP, PyTorch)."
    )
    parser.add_argument('--csv', type=str, required=True,
                        help="Path al CSV con i dati registrati dal pilota esperto.")
    parser.add_argument('--out-dir', type=str, default='./output',
                        help="Directory in cui salvare modello e scaler.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed per riproducibilita'.")
    return parser.parse_args()
    

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INIT] Device: {device}")
    if device.type == 'cuda':
        print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")

    model_path  = os.path.join(args.out_dir, 'driving_mlp_best.pth')
    scaler_path = os.path.join(args.out_dir, 'scaler.joblib')


    X, y = load_and_preprocess(args.csv, scaler_path)
    train_loader, val_loader = build_dataloaders(X, y, args.seed, device)

    model     = DrivingMLP(input_dim=INPUT_DIM).to(device)
    criterion = WeightedMSELoss(LOSS_WEIGHTS).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] DrivingMLP creato  |  Parametri allenabili: {n_params:,}")
    print(f"[MODEL] Loss weights -> steer={STEER_WEIGHT}  "
          f"accel={PEDAL_WEIGHT}  brake={PEDAL_WEIGHT}")
    print(f"[TRAIN] Max epochs: {MAX_EPOCHS}  |  Patience: {EARLY_STOP_PAT}  "
          f"|  Batch size: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print("-" * 78)

    best_val_loss     = float('inf')
    epochs_no_improve = 0
    best_epoch        = 0
    t_start           = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        dt         = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss     = val_loss
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            tag = "  <-- BEST (saved)"
        else:
            epochs_no_improve += 1
            tag = f"  (no improve: {epochs_no_improve}/{EARLY_STOP_PAT})"

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  "
              f"|  Train Loss: {train_loss:.6f}  "
              f"|  Val Loss: {val_loss:.6f}  "
              f"|  {dt:5.1f}s{tag}")

    
        if epochs_no_improve >= EARLY_STOP_PAT:
            print(f"\n[STOP] Early stopping: nessun miglioramento "
                  f"per {EARLY_STOP_PAT} epoche consecutive.")
            break

    total_time = time.time() - t_start
    print("-" * 78)
    print(f"[DONE] Training terminato in {total_time:.1f}s "
          f"({total_time/60:.1f} min)")
    print(f"[DONE] Best Validation Loss: {best_val_loss:.6f}  (epoch {best_epoch})")
    print(f"[DONE] Modello salvato in: {model_path}")
    print(f"[DONE] Scaler salvato in:  {scaler_path}")


if __name__ == '__main__':
    main()