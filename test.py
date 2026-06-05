import sys
import torch
sys.path.insert(0,'test_suites')
from _hamiltonian import DoublePendulum
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np


from transformer_model import Config, ParamInferenceTransformer

cfg= Config(n_head=16,embed_dim=256,hidden_dim=2048,data_len=5000)
model_transformer=ParamInferenceTransformer(cfg)

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_transformer.eval()
model_transformer.load_state_dict(torch.load('outputs/model_dlen5000_big.pth',map_location=dev))

## loading the dataset
dataset_inference_test= pd.read_parquet('datasets/dataset_doublependulumpts_finetune.parquet')
config_groups=dataset_inference_test.groupby('config_id')
trajectories={cid: group.copy() for cid,group in config_groups}

traj_ids=list(trajectories.keys())
train_trajectories=torch.tensor(np.array([trajectories[i] for i in traj_ids]),dtype=torch.float32)

norm_trajectories=[]
param_storage=[]

for i in range(len(train_trajectories)):
    angle_pendulum_1_norm=torch.sin(train_trajectories[i][:,6])
    angle_pendulum_2_norm=torch.sin(train_trajectories[i][:,7])
    omega_1_norm=torch.sin(train_trajectories[i][:,8])
    omega_2_norm=torch.sin(train_trajectories[i][:,9])
    mass_1,mass_2,length_1,length_2=train_trajectories[i][0,1],train_trajectories[i][0,2],train_trajectories[i][0,3],train_trajectories[i][0,4]
    parameters=torch.tensor([mass_1,mass_2,length_1,length_2])
    param_storage.append(parameters) 
    new=torch.stack((angle_pendulum_1_norm,angle_pendulum_2_norm,omega_1_norm,omega_2_norm),dim=-1)
    norm_trajectories.append(new)
   
param_tensor = torch.stack(param_storage, dim=0).to(device=dev)
norm_trajectories=torch.stack(norm_trajectories,dim=0).to(device=dev)
class DoublePendulumData(Dataset):
    def __init__(self, params, trajs):
        self.params = params
        self.trajectories = trajs
    def __len__(self):
        return len(self.params)
    def __getitem__(self, idx):
        return self.params[idx], self.trajectories[idx]

loader = DataLoader(DoublePendulumData(param_tensor, norm_trajectories), batch_size=32)

with torch.no_grad():
    u=0
    for t_tensor, traj in loader:
        pred_params=model_transformer(traj)
        print(pred_params)
        u+=1
        if u>=1:
            break
        

