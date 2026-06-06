import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader
import time
from transformer_model import Config,ParamInferenceTransformer
import sys
sys.path.insert(0,'test_suites')
from _hamiltonian import DoublePendulum

@dataclass
class config_transformer:
    batch_size:int=int(os.getenv("BATCH",64))
    lr:float=float(os.getenv("LR",1e-4))
    num_epochs:int=int(os.getenv("NE",1))
    lambda:float=float(os.getenv("LAMBDA",0.54))

transformer_setup=config_transformer()
    
parser = argparse.ArgumentParser(description="Train Transformer on double pendulum data")

# Output / model
parser.add_argument("--model_name", type=str, default=os.getenv("NAME", "model_file.pth"), help="Name of saved model file")
parser.add_argument("--model_type", type=str, default=os.getenv("TYPE", "transformer"), help="Model Type")
# Paths
parser.add_argument("--dataset_dir", type=str, default=os.getenv("DATASET_DIR", "datasets"), help="Dataset folder path")
parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR", "outputs"), help="Output folder path")

args = parser.parse_args()

trainingdata=pd.read_parquet('datasets/dataset_doublependulumpts_finetune.parquet')
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
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

norm_trajectories=[]
param_storage=[]
true_trajectories=[]

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

    theta1=train_trajectories[i][:,10]
    theta2=train_trajectories[i][:,11]
    omega1=train_trajectories[i][:,12]
    omega2=train_trajectories[i][:,13]
    true_tra=torch.stack((theta1,theta2,omega1,omega2),dim=-1)
    true_trajectories.append(true_tra)
  
param_tensor_train = torch.stack(param_storage, dim=0).to(device=dev)
norm_trajectories_train=torch.stack(norm_trajectories,dim=0).to(device=dev)
true_trajectories_train=torch.stack(true_trajectories,dim=0).to(device=dev)

print(param_tensor_train.shape)
print(norm_trajectories_train.shape)
print(true_trajectories_train.shape)

# catering data to torch emthod for transformer
class DoublePendulumData(Dataset):
    def __init__(self,param_tensor,norm_trajectories, noiseless_traj):
        self.params=param_tensor
        self.trajectories=norm_trajectories
        self.noiseless_traj=noiseless_traj
    def __len__(self):
        return len(self.params)
    def __getitem__(self, index):
        traj=self.trajectories[index]
        target_params=self.params[index]
        true_trajecs=self.noiseless_traj
        return target_params,traj,true_trajecs
   
data=DataLoader(DoublePendulumData(param_tensor_train,norm_trajectories_train,true_trajectories_train),batch_size=transformer_setup.batch_size,shuffle=True)

for target_params, traj in data:
    print("Batch target params:", target_params.shape)  # -> [16, 4]
    print("Batch trajectories:", traj.shape)            # -> [16, 5000, 4]
    break  

## training the model
cfg=Config(n_head=16,embed_dim=256,hidden_dim=2048,data_len=5000)
model_transformer=ParamInferenceTransformer(cfg)
model_transformer.load_state_dict(torch.load('outputs/model_dlen5000_big.pth',map_location=dev))
model_transformer.to(dev)

print(f'training {args.model_type}')

obj_func=MSELoss()
optimizer=Adam(model_transformer.parameters(),lr=transformer_setup.lr)

start_time=time.time()
loss_hist=[]
iter_number=[]
for epoch in range(transformer_setup.num_epochs):
    epoch_loss = 0
    for target_params, traj, noiseless in data:
        optimizer.zero_grad()
        pred_params = model_tranformer(traj)
        
        h_loss=0
        for i in range(transformer_setup.batch_size):
            dp = DoublePendulum(*pred_params[i].tolist())
            H = np.array([dp.hamiltonian(*s) for s in noiseless[i].numpy()])
            h_loss += np.var(H) / np.mean(np.abs(H))
        h_loss/=transformer_setup.batch_size
        loss = obj_func(pred_params, target_params)+h_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(data)  # average over batches
    print(f"Epoch {epoch}: avg_loss = {avg_loss:.6f}")
    loss_hist.append(avg_loss)
    iter_number.append(epoch)

total_runtime=time.time()-start_time
print(f'total runtime:{total_runtime:.3f}')

os.makedirs("outputs", exist_ok=True)
plt.plot(iter_number,loss_hist)
plt.xlabel('Iteration #')
plt.ylabel('Loss')
plt.title('Training loss')
plt.savefig(f"outputs/loss_plot_{args.model_name.replace('.pth','')}.png")
plt.close()
torch.save(model_transformer.state_dict(),f"outputs/{args.model_name}")