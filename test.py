import sys
import torch
sys.path.insert(0,'test_suites')
from _hamiltonian import DoublePendulum
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import time
from transformer_model import Config, ParamInferenceTransformer

cfg= Config(n_head=16,embed_dim=256,hidden_dim=2048,data_len=5000)
model_transformer=ParamInferenceTransformer(cfg)

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_transformer.eval()
model_transformer.load_state_dict(torch.load('outputs/model_dlen5000_big.pth',map_location=dev))

## loading the dataset
dataset_inference_test= pd.read_parquet('datasets/dataset_doublependulumpts_finetune.parquet')
h_losses = []
u=0
start_time=time.time()
for cid, group in dataset_inference_test.groupby("config_id"):
    traj = torch.tensor(group.values, dtype=torch.float32)
    
    # Model input
    x = torch.stack([
        torch.sin(traj[:, 6]), torch.sin(traj[:, 7]),
        traj[:, 8], traj[:, 9],
    ], dim=-1).unsqueeze(0)
    
    with torch.no_grad():
        pred = model_transformer(x).squeeze(0)
    
    # H-loss on noiseless states
    states = traj[:, 10:14].numpy()
    m1, m2, l1, l2 = pred.tolist()
    dp = DoublePendulum(m1, m2, l1, l2)
    H = np.array([dp.hamiltonian(s[0], s[1], s[2], s[3]) for s in states])
    h_loss = np.var(H) / np.mean(np.abs(H))
    h_losses.append(h_loss)
    u+=1
    print(f'steps completed: {u}')
    if u>=50:
        break

time_taken=time.time()-start_time
print(f'time taken: {time_taken} \n')
print(f"Mean H-loss over Set B: {np.mean(h_losses):.6e}")        
        

