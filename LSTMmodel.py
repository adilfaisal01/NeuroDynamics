import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    num_layers:int=3
    hidden_size:int=512
    batch_first:bool=True
    dropout:float=0.05
    input_features: int=4

class parameterEstimationLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.lstm=nn.LSTM(
            input_size=config.input_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=config.batch_first
        )

        self.regression= nn.Sequential(
            nn.Linear(in_features=config.hidden_size,out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=config.input_features)
            
        )
    def forward (self,x):
        output,(hn,cn)=self.lstm(x)
        last_hidden=hn[-1]

        return self.regression(last_hidden)



