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

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/outputs")

@dataclass
class config_transformer:
    batch_size:int=int(os.getenv("BATCH",1))
    lr:float=float(os.getenv("LR",1e-4))
    num_epochs:int=int(os.getenv("NE",1))
    lambda_h:float=float(os.getenv("LAMBDA",0.5))

def hamiltonian_loss(pred_params, states, g=9.81):
    m1, m2, l1, l2 = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3]
    th1, om1, th2, om2 = states[:, :, 0], states[:, :, 1], states[:, :, 2], states[:, :, 3]
    
    V = -(m1 + m2).unsqueeze(1) * g * l1.unsqueeze(1) * torch.cos(th1) - m2.unsqueeze(1) * g * l2.unsqueeze(1) * torch.cos(th2)
    T1 = 0.5 * m1.unsqueeze(1) * l1.unsqueeze(1)**2 * om1**2
    T2 = 0.5 * m2.unsqueeze(1) * (l1.unsqueeze(1)**2 * om1**2 + l2.unsqueeze(1)**2 * om2**2 + 2 * l1.unsqueeze(1) * l2.unsqueeze(1) * om1 * om2 * torch.cos(th1 - th2))
    
    H = T1 + T2 + V                              # (B, T)
    H_var = torch.var(H, dim=1)                   # (B,)
    H_mean = torch.mean(torch.abs(H), dim=1)      # (B,)
    return H_var / H_mean                         # (B,)

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
        true_trajecs=self.noiseless_traj[index]
        return target_params,traj,true_trajecs
   
data=DataLoader(DoublePendulumData(param_tensor_train,norm_trajectories_train,true_trajectories_train),batch_size=transformer_setup.batch_size,shuffle=True)

for target_params, traj, _ in data:
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

def plot_loss_curves(mse_hist, ham_hist, combined_hist, epochs, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, mse_hist, color='blue', label='MSE Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('MSE Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, ham_hist, color='red', label='Hamiltonian Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Hamiltonian Loss')
    axes[1].set_title('Hamiltonian Loss')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, mse_hist, color='blue', label='MSE Loss')
    axes[2].plot(epochs, ham_hist, color='red', label='Hamiltonian Loss')
    axes[2].plot(epochs, combined_hist, color='green', label='Combined Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('All Loss Terms')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

start_time=time.time()
mse_hist=[]
ham_hist=[]
combined_hist=[]
iter_number=[]
for epoch in range(transformer_setup.num_epochs):
    epoch_mse = 0
    epoch_ham = 0
    epoch_combined = 0
    for target_params, traj, noiseless in data:
        optimizer.zero_grad()
        pred_params = model_transformer(traj)
        h_loss=hamiltonian_loss(pred_params,noiseless).mean()
        mse_loss = obj_func(pred_params, target_params)
        loss = mse_loss+transformer_setup.lambda_h*h_loss
        loss.backward()
        optimizer.step()
        epoch_mse += mse_loss.item()
        epoch_ham += h_loss.item()
        epoch_combined += loss.item()
    
    avg_mse = epoch_mse / len(data)
    avg_ham = epoch_ham / len(data)
    avg_combined = epoch_combined / len(data)
    print(f"Epoch {epoch}: MSE={avg_mse:.6f}, Ham={avg_ham:.6f}, Combined={avg_combined:.6f}")
    mse_hist.append(avg_mse)
    ham_hist.append(avg_ham)
    combined_hist.append(avg_combined)
    iter_number.append(epoch)
    if (epoch + 1) % 5 == 0:
        ckpt_name = args.model_name.replace('.pth', f'_epoch{epoch+1}.pth')
        torch.save(model_transformer.state_dict(), f"{args.output_dir}/{ckpt_name}")
        print(f"  Checkpoint saved: {ckpt_name}")

total_runtime=time.time()-start_time
print(f'total runtime:{total_runtime:.3f}')

os.makedirs(args.output_dir, exist_ok=True)
plot_loss_curves(mse_hist, ham_hist, combined_hist, iter_number,
                 f"{args.output_dir}/loss_plot_{args.model_name.replace('.pth','')}.png")
torch.save(model_transformer.state_dict(),f"{args.output_dir}/{args.model_name}")

norm_trajectories_test=[]
param_storage_test=[]
true_trajectories_test=[]

for i in range(len(test_trajectories)):
    angle_pendulum_1_norm=torch.sin(test_trajectories[i][:,6])
    angle_pendulum_2_norm=torch.sin(test_trajectories[i][:,7])
    omega_1_norm=torch.sin(test_trajectories[i][:,8])
    omega_2_norm=torch.sin(test_trajectories[i][:,9])
    mass_1,mass_2,length_1,length_2=test_trajectories[i][0,1],test_trajectories[i][0,2],test_trajectories[i][0,3],test_trajectories[i][0,4]
    parameters=torch.tensor([mass_1,mass_2,length_1,length_2])
    param_storage_test.append(parameters)
    new=torch.stack((angle_pendulum_1_norm,angle_pendulum_2_norm,omega_1_norm,omega_2_norm),dim=-1)
    norm_trajectories_test.append(new)

    theta1=test_trajectories[i][:,10]
    theta2=test_trajectories[i][:,11]
    omega1=test_trajectories[i][:,12]
    omega2=test_trajectories[i][:,13]
    true_tra=torch.stack((theta1,theta2,omega1,omega2),dim=-1)
    true_trajectories_test.append(true_tra)

param_tensor_test = torch.stack(param_storage_test, dim=0).to(device=dev)
norm_trajectories_test=torch.stack(norm_trajectories_test,dim=0).to(device=dev)
true_trajectories_test=torch.stack(true_trajectories_test,dim=0).to(device=dev)

print(param_tensor_test.shape)
print(norm_trajectories_test.shape)

test_data=DataLoader(DoublePendulumData(param_tensor_test,norm_trajectories_test,true_trajectories_test))

model_transformer.eval()
all_preds=[]
all_targets=[]
loss_test=[]
hamiltonian_loss_values=[]
with torch.no_grad():
    test_loss=0
    h_loss_total=0
    for target_params,traj,noiseless in test_data:
        predicted_params=model_transformer(traj)
        mse=obj_func(predicted_params,target_params)
        test_loss+=mse.item()
        h_val=hamiltonian_loss(predicted_params,noiseless).mean()
        h_loss_total+=h_val.item()
        all_targets.append(target_params.cpu())
        all_preds.append(predicted_params.cpu())
    avg_test_loss=test_loss/len(test_data)
    avg_h_loss=h_loss_total/len(test_data)
    loss_test.append(avg_test_loss)
    hamiltonian_loss_values.append(avg_h_loss)

print(f'Test loss (MSE): {np.mean(loss_test):.6f}')
print(f'Test loss (Hamiltonian): {np.mean(hamiltonian_loss_values):.6f}')

all_targets = torch.cat(all_targets, dim=0).numpy()
all_preds = torch.cat(all_preds, dim=0).numpy()

errors=all_preds-all_targets
for i in range(errors.shape[1]):
    plt.subplot(1, errors.shape[1], i+1)
    plt.plot(errors[:, i], marker='o', linestyle='', alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Sample Index")
    plt.ylabel("Error")
    plt.title(f"Error for Param {i+1}")

plt.tight_layout()
plt.savefig(f"{args.output_dir}/true_vs_predicted_{args.model_name.replace('.pth','')}_test_mse={avg_test_loss:.6f}_ham={avg_h_loss:.6f}.png")
plt.close()

average_errors=errors.mean(axis=0)
error_var=np.var(errors,axis=0)
error_df=pd.DataFrame({
    "param_index": list(range(errors.shape[1])),
    "avg_error": average_errors,
    "Variance":error_var
})
csv_path=f"{args.output_dir}/true_vs_predicted_{args.model_name.replace('.pth','')}_test_mse={avg_test_loss:.6f}_ham={avg_h_loss:.6f}.csv"
error_df.to_csv(csv_path,index=False)