import sys
import torch
sys.path.insert(0,'test_suites')
from _hamiltonian import DoublePendulum
from torch.utils.data import Dataset,DataLoader
import pandas as pd


from transformer_model import Config, ParamInferenceTransformer

cfg= Config(n_head=16,embed_dim=256,hidden_dim=2048,data_len=5000)
model_transformer=ParamInferenceTransformer(cfg)

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_transformer.eval()
model_transformer.load_state_dict(torch.load('outputs/model_dlen5000_big.pth',map_location=dev))

## loading the dataset
dataset_inference_test= pd.read_parquet('datasets/dataset_doublependulum_finetune.parquet')

