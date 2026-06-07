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
model_2=ParamInferenceTransformer(cfg)

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_transformer.eval()
model_2.eval()
model_transformer.load_state_dict(torch.load('outputs/model_dlenfinetune_3e-5_0.4_big.pth',map_location=dev))
model_2.load_state_dict(torch.load('outputs/model_dlen5000_big.pth',map_location=dev))
## loading the dataset
dataset_inference_test= pd.read_parquet('datasets/dataset_doublependulumpts_setC.parquet')


alpha_lis  =[0.1,0.3,0.5,0.7,0.9]
start_time=time.time()
for alpha in alpha_lis:
    u=0
    h_losses = []
    param_losses=[]
    for cid, group in dataset_inference_test.groupby("config_id"):
        traj = torch.tensor(group.values, dtype=torch.float32)
        
        # Model input
        x = torch.stack([
            torch.sin(traj[:, 6]), torch.sin(traj[:, 7]),
            torch.sin(traj[:, 8]), torch.sin(traj[:, 9]),
        ], dim=-1).unsqueeze(0)
        
        with torch.no_grad():
            pred1 = model_transformer(x).squeeze(0)
            pred2= model_2(x).squeeze(0)
        pred=(alpha*pred1+(1-alpha)*pred2)
        # True params (constant across trajectory, first row)
        true_params = traj[0, 1:5]  # m1, m2, l1, l2
        # MSE
        mse = ((pred - true_params) ** 2)
        param_losses.append(mse)
        # H-loss on noiseless states
        states = traj[:, 10:14].numpy()
        m1, m2, l1, l2 = pred.tolist()
        dp = DoublePendulum(m1, m2, l1, l2)
        H = np.array([dp.hamiltonian(s[0], s[1], s[2], s[3]) for s in states])
        h_loss = np.var(H) / np.mean(np.abs(H))
        h_losses.append(h_loss)
        u+=1
        
        if u>=50:
            print(f'steps completed: {u}')
            break
    print(f"Mean H-loss over Set C_{alpha}: {np.mean(h_losses):.6e} \n")
    print(f"mean param loss over setc C_{alpha}: {np.mean(param_losses)}\n")

time_taken=time.time()-start_time
print(f'time taken: {time_taken} \n')
# print(f"Mean H-loss over Set C: {np.mean(h_losses):.6e} \n")
# print(f"mean param loss over setc C: {np.mean(param_losses)}\n")
print()
        

