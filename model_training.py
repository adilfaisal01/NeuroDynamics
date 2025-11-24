import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch


trainingdata=pd.read_csv('dataset_doublependulum.csv')
config_groups=trainingdata.groupby('config_id')
trajectories={cid: group.copy() for cid,group in config_groups}

traj_ids=list(trajectories.keys())

# splitting trajectories 
train_id, test_id=train_test_split(
    traj_ids,
    random_state=100,
    shuffle=True
)
train_trajectories=torch.tensor(np.array([trajectories[i] for i in train_id]),dtype=torch.float32)
test_trajectories=torch.tensor(np.array([trajectories[i] for i in test_id]),dtype=torch.float32)


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
    new=torch.stack((angle_pendulum_1_norm,angle_pendulum_2_norm,omega_1_norm,omega_2_norm),axis=-1)
    norm_trajectories.append(new)
   

param_tensor = torch.stack(param_storage, dim=0)
norm_trajectories=torch.stack(norm_trajectories,dim=0)

print(param_tensor.shape)
print(norm_trajectories.shape)

# catering data to torch emthod for transformer

from torch.utils.data import Dataset,DataLoader

class DoublePendulumData(Dataset):
    def __init__(self,param_tensor,norm_trajectories):
        self.params=param_tensor
        self.trajectories=norm_trajectories
    def __len__(self):
        return len(self.params)
    def __getitem__(self, index):
        traj=self.trajectories[index]
        target_params=self.params[index]
        return target_params,traj
   
data=DataLoader(DoublePendulumData(param_tensor,norm_trajectories),batch_size=32,shuffle=True)

for target_params, traj in data:
    print("Batch target params:", target_params.shape)  # -> [16, 4]
    print("Batch trajectories:", traj.shape)            # -> [16, 5000, 4]
    break  

from torch.nn import MSELoss
from torch.optim import Adam

from transformer_model import Config,ParamInferenceTransformer

modelconfig=Config(n_head=2,embed_dim=16,hidden_dim=32)
model=ParamInferenceTransformer(modelconfig)

obj_func=MSELoss()
optimizer=Adam(model.parameters(),lr=1e-4)
num_epochs=1
for epoch in range(num_epochs):
    epoch_loss = 0
    for target_params, traj in data:
        optimizer.zero_grad()
        pred_params = model(traj)
        loss = obj_func(pred_params, target_params)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(data)  # average over batches
    print(f"Epoch {epoch}: avg_loss = {avg_loss:.6f}")







        

