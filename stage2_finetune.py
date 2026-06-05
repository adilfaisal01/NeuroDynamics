"""
Stage 2 fine-tuning: MSE + Hamiltonian loss refinement.

Loads Stage 1 model (dlen5000), fine-tunes on Set B (3000 configs)
with combined loss: L = MSE + lambda * H_loss

H_loss: variance of H(t; pred_params) along the observed noiseless trajectory.
If params are correct, energy is conserved (low variance).
If params are wrong, H(t) fluctuates (high variance).
"""
import sys
sys.path.insert(0, '/mnt/E/github-projects/NeuroDynamics/test_suites')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from _hamiltonian import DoublePendulum
import time
import os
import math


# --- Config ---
MODEL_PATH = "/mnt/E/github-projects/NeuroDynamics/outputs/model_dlen5000_big.pth"
FINE_TUNE_PATH = "/mnt/E/github-projects/NeuroDynamics/datasets/dataset_doublependulumpts_finetune.parquet"
OUTPUT_DIR = "/mnt/E/github-projects/NeuroDynamics/stage2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_EPOCHS = 50
LR = 1e-5          # Low LR — we're refining, not retraining
LAMBDA_H = 0.1      # Weight for Hamiltonian loss
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Exact Stage 1 model architecture (from transformer_model.py) ---
class PositionalEncoding(nn.Module):
    def __init__(self, data_len=5000, embed_dim=256, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(data_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(data_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class ParameterTransformer(nn.Module):
    def __init__(self, input_dim=4, embed_dim=256, hidden_dim=2048, nhead=8, num_layers=3,
                 data_len=5000, output_dim=4):
        super().__init__()
        # Downsampling: kernel = 5000 // data_len (identity when data_len=5000)
        self.downsampling = nn.AvgPool1d(kernel_size=5000 // data_len)

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.position_encoding = PositionalEncoding(data_len, embed_dim, dropout=0.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.regression = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # x: (B, T, 4) — sin-transformed states
        x = x.transpose(1, 2)       # (B, 4, T) for AvgPool1d
        x = self.downsampling(x)    # (B, 4, T')
        x = x.transpose(1, 2)       # (B, T', 4)

        x = self.input_proj(x)      # (B, T', embed_dim)
        x = self.position_encoding(x)
        x = self.encoder(x)         # (B, T', embed_dim)

        x = x.transpose(1, 2)       # (B, embed_dim, T') for AdaptiveAvgPool1d
        x = self.pool(x)            # (B, embed_dim, 1)
        x = x.squeeze(-1)           # (B, embed_dim)

        return self.regression(x)   # (B, 4) — [m1, m2, l1, l2]


# --- Dataset: loads trajectories, applies sin-transform for model input ---
class PendulumFineTuneDataset(Dataset):
    def __init__(self, parquet_path, max_samples=None):
        self.df = pd.read_parquet(parquet_path)
        self.config_ids = sorted(self.df['config_id'].unique())
        if max_samples:
            self.config_ids = self.config_ids[:max_samples]

    def __len__(self):
        return len(self.config_ids)

    def __getitem__(self, idx):
        cid = self.config_ids[idx]
        traj = self.df[self.df['config_id'] == cid].sort_values('time')

        # Raw noiseless states for H-loss: (T, 4) [theta1, omega1, theta2, omega2]
        raw_states = np.column_stack([
            traj['no noise angle pendulum 1'].values,
            traj['no noise angularvel pendulum 1'].values,
            traj['no noise angle pendulum 2'].values,
            traj['no noise angularvel pendulum 2'].values,
        ]).astype(np.float32)

        # Sin-transformed states for model input (Stage 1 was trained on sin of noiseless)
        model_states = np.sin(raw_states).astype(np.float32)

        # True params
        params = np.array([
            traj['mass pendulum 1'].iloc[0],
            traj['mass pendulum 2'].iloc[0],
            traj['length pendulum 1'].iloc[0],
            traj['length pendulum 2'].iloc[0],
        ], dtype=np.float32)

        return torch.from_numpy(model_states), torch.from_numpy(raw_states), torch.from_numpy(params)


# --- Hamiltonian loss (batched over a batch of trajectories) ---
def hamiltonian_loss_batch(pred_params_batch, states_batch):
    """
    pred_params_batch: (B, 4) tensor — predicted [m1, m2, l1, l2]
    states_batch: (B, T, 4) tensor — noiseless states
    Returns: (B,) tensor — H_loss for each trajectory in batch
    """
    B, T, _ = states_batch.shape
    losses = []

    for b in range(B):
        m1, m2, l1, l2 = pred_params_batch[b].detach().cpu().numpy()
        dp = DoublePendulum(float(m1), float(m2), float(l1), float(l2))

        H_vals = []
        for t in range(T):
            s = states_batch[b, t]
            H = dp.hamiltonian(float(s[0]), float(s[1]), float(s[2]), float(s[3]))
            H_vals.append(H)

        H = np.array(H_vals)
        H_mean = np.mean(np.abs(H))
        if H_mean < 1e-10:
            losses.append(0.0)
        else:
            losses.append(float(np.var(H) / H_mean))

    return torch.tensor(losses, device=pred_params_batch.device, dtype=pred_params_batch.dtype)


# --- Main training loop ---
def main():
    print(f"Device: {DEVICE}")
    print(f"Loading model from {MODEL_PATH}")

    model = ParameterTransformer(
        embed_dim=256, hidden_dim=2048, nhead=8, num_layers=3, data_len=5000
    ).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    # Handle potential key mismatch (state dict may have 'module.' prefix)
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    print(f"Loaded Stage 1 model ({sum(p.numel() for p in model.parameters()):,} params)")

    dataset = PendulumFineTuneDataset(FINE_TUNE_PATH)
    print(f"Fine-tune dataset: {len(dataset)} trajectories")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    mse_loss_fn = nn.MSELoss()

    # Freeze all but regression head initially
    for name, param in model.named_parameters():
        if 'regression' not in name:
            param.requires_grad = False

    print("Training regression head only (frozen transformer)...")

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        total_mse = 0.0
        total_h = 0.0
        total_loss = 0.0
        n_batches = 0

        # After 10 epochs, unfreeze everything
        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
            print("Unfrozen all layers for full fine-tuning...")

        model.train()
        for model_states, raw_states, true_params in dataloader:
            model_states = model_states.to(DEVICE)
            raw_states = raw_states.to(DEVICE)
            true_params = true_params.to(DEVICE)

            pred_params = model(model_states)

            mse_loss = mse_loss_fn(pred_params, true_params)

            # H-loss: compute on raw (non-sin) states using predicted params
            h_loss = hamiltonian_loss_batch(pred_params, raw_states)
            h_loss = h_loss.mean()

            # Combined loss
            loss = mse_loss + LAMBDA_H * h_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_mse += mse_loss.item()
            total_h += h_loss.item()
            total_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - epoch_start
        avg_mse = total_mse / n_batches
        avg_h = total_h / n_batches
        avg_loss = total_loss / n_batches

        # Quick eval on a few test trajectories
        test_h_loss = 0.0
        test_mse_loss = 0.0
        model.eval()
        with torch.no_grad():
            n_test = 0
            for model_states, raw_states, true_params in dataloader:
                if n_test >= 3:
                    break
                model_states = model_states[:1].to(DEVICE)
                raw_states = raw_states[:1].to(DEVICE)
                true_params = true_params[:1].to(DEVICE)

                pred = model(model_states)
                test_mse_loss += mse_loss_fn(pred, true_params).item()
                hl = hamiltonian_loss_batch(pred, raw_states)
                test_h_loss += hl.mean().item()
                n_test += 1
        test_mse_loss /= n_test
        test_h_loss /= n_test

        print(f"Epoch {epoch+1:2d}/{N_EPOCHS} | "
              f"MSE={avg_mse:.6f} | H_loss={avg_h:.6e} | "
              f"Total={avg_loss:.6f} | "
              f"Test MSE={test_mse_loss:.6f} Test H={test_h_loss:.6e} | "
              f"Time={epoch_time:.1f}s")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"model_stage2_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = os.path.join(OUTPUT_DIR, "model_stage2_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nDone! Final model saved to {final_path}")


if __name__ == "__main__":
    main()