import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description="Train Transformer on double pendulum data")

# Embed and model dimensions
parser.add_argument("--embed_dim", type=int, default=int(os.getenv("ED", 16)), help="Embedding dimension")
parser.add_argument("--hidden_dim", type=int, default=int(os.getenv("HD", 32)), help="Feedforward hidden dimension")
parser.add_argument("--n_head", type=int, default=int(os.getenv("HEAD", 2)), help="Number of attention heads")

# Training settings
parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH", 32)), help="Batch size")
parser.add_argument("--lr", type=float, default=float(os.getenv("LR", 1e-4)), help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=int(os.getenv("NE", 1)), help="Number of training epochs")
parser.add_argument("--dseq_len", type=int, default=int(os.getenv("DS_LEN", 1000)), help="Downsampled length of trajectory if any")

# Output / model
parser.add_argument("--model_name", type=str, default=os.getenv("NAME", "model_file.pth"), help="Name of saved model file")
parser.add_argument("--model_type", type=str, default=os.getenv("TYPE", "model_file.pth"), help="Model Type")


# Paths
parser.add_argument("--dataset_dir", type=str, default=os.getenv("DATASET_DIR", "datasets"), help="Dataset folder path")
parser.add_argument("--output_dir", type=str, default=os.getenv("OUTPUT_DIR", "outputs"), help="Output folder path")

args = parser.parse_args()

trainingdata=pd.read_csv('datasets/dataset_doublependulum_22.csv')
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
   
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
param_tensor_train = torch.stack(param_storage, dim=0).to(device=dev)
norm_trajectories_train=torch.stack(norm_trajectories,dim=0).to(device=dev)

print(param_tensor_train.shape)
print(norm_trajectories_train.shape)

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
   
data=DataLoader(DoublePendulumData(param_tensor_train,norm_trajectories_train),batch_size=args.batch_size,shuffle=True)

for target_params, traj in data:
    print("Batch target params:", target_params.shape)  # -> [16, 4]
    print("Batch trajectories:", traj.shape)            # -> [16, 5000, 4]
    break  

from torch.nn import MSELoss
from torch.optim import Adam


if args.model_type=="transformer":
    from transformer_model import Config,ParamInferenceTransformer
    ## training the model
    modelconfig=Config(n_head=args.n_head,embed_dim=args.embed_dim,hidden_dim=args.hidden_dim,data_len=args.dseq_len)
    model=ParamInferenceTransformer(modelconfig).to(dev)
elif args.model_type=="lstm":
    from LSTMmodel import Config, parameterEstimationLSTM
    modelconfig=Config(hidden_size=args.hidden_dim)
    model=parameterEstimationLSTM(modelconfig).to(dev)
else:
    raise ValueError("Please clarify which model to be used for training and testing")

print(f'training {args.model_type}')

obj_func=MSELoss()
optimizer=Adam(model.parameters(),lr=args.lr)
import time
start_time=time.time()
loss_hist=[]
iter_number=[]
for epoch in range(args.num_epochs):
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
torch.save(model.state_dict(),f"outputs/{args.model_name}")

## Processing testing data 

norm_trajectories_test=[]
param_storage_test=[]

for i in range(len(test_trajectories)):
    angle_pendulum_1_norm=torch.sin(test_trajectories[i][:,6])
    angle_pendulum_2_norm=torch.sin(test_trajectories[i][:,7])
    omega_1_norm=torch.sin(test_trajectories[i][:,8])
    omega_2_norm=torch.sin(test_trajectories[i][:,9])
    mass_1,mass_2,length_1,length_2=test_trajectories[i][0,1],test_trajectories[i][0,2],test_trajectories[i][0,3],test_trajectories[i][0,4]
    parameters=torch.tensor([mass_1,mass_2,length_1,length_2])
    param_storage_test.append(parameters) 
    new=torch.stack((angle_pendulum_1_norm,angle_pendulum_2_norm,omega_1_norm,omega_2_norm),axis=-1)
    norm_trajectories_test.append(new)
   
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
param_tensor_test = torch.stack(param_storage_test, dim=0).to(device=dev)
norm_trajectories_test=torch.stack(norm_trajectories_test,dim=0).to(device=dev)

print(param_tensor_test.shape)
print(norm_trajectories_test.shape)

test_data=DataLoader(DoublePendulumData(
    param_tensor=param_tensor_test,
    norm_trajectories=norm_trajectories_test
))

model.eval()
all_preds=[]
all_targets=[]
loss_test=[]
with torch.no_grad():
    test_loss=0
    for target_params,traj in test_data:
        predicted_params=model(traj)
        loss=obj_func(predicted_params,target_params)
        test_loss=test_loss+loss.item()
        all_targets.append(target_params.cpu())
        all_preds.append(predicted_params.cpu())
        avg_test_loss=test_loss/len(test_data)
    loss_test.append(avg_test_loss)

print(f'Test loss (MSE): {np.mean(loss_test)}')

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
plt.savefig(f"outputs/true_vs_predicted {args.model_name} test error={avg_test_loss} run_time(s)={total_runtime} model {args.model_type}.png")
# plt.show()

## creating a csv of error files
average_errors=errors.mean(axis=0)
error_var=np.var(errors,axis=0)
error_df=pd.DataFrame({
    "param_index": list(range(errors.shape[1])),
    "avg_error": average_errors,
    "Variance":error_var 
})
csv_path=f"outputs/true_vs_predicted {args.model_name} test error={avg_test_loss} run_time(s)={total_runtime} model {args.model_type}.csv"
error_df.to_csv(csv_path,index=False)


        

