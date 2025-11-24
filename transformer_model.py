### Transformer Model
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    input_dim=4
    dropout:float=0.05
    embed_dim:int=256
    num_layers:int=3
    hidden_dim:int=1024
    n_head:int=32
    bias:bool=True
    num_params:int=4


class ParamInferenceTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj=nn.Linear(config.input_dim,config.embed_dim)
        transformerencoderlayer=nn.TransformerEncoderLayer(d_model=config.embed_dim,
                                                    nhead=config.n_head,
                                                    dropout=config.dropout,
                                                    dim_feedforward=config.hidden_dim,
                                                    batch_first=True
                                                    )
        self.encoder=nn.TransformerEncoder(num_layers=config.num_layers,encoder_layer=transformerencoderlayer)
        self.pool=nn.AvgPool1d(kernel_size=5000)
        self.regression=nn.Sequential(
            nn.Linear(in_features=config.embed_dim,out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=config.num_params)
        )
    
    def forward(self,x):
        x=self.input_proj(x)

        x=self.encoder(x)

        x=x.transpose(1,2)

        x=self.pool(x)

        x=x.squeeze(-1)

        out=self.regression(x)

        return out
