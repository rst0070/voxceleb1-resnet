# trainer.py
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import TrainDataset
from arguments import get_args
import wandb

sys_args, exp_args = get_args()

GPU = sys_args['gpu']
NUM_WORKERS = sys_args['num_workers']

class Trainer():
    def __init__(self, model, dataset:TrainDataset, optimizer, batch_size):
        
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = NUM_WORKERS)
        # 아래는 single class classification
        self.loss_function = nn.CrossEntropyLoss().to(GPU)
        # 아래는 multi label classification
        # self.loss_function = nn.MultiLabelSoftMarginLoss().to(GPU)
        self.optimizer = optimizer

    def train(self):
        self.model.train()
        for X, y in tqdm(self.dataloader, desc = "training"):
            
            self.optimizer.zero_grad()
            
            X = X.to(GPU)
            y = y.to(GPU)
            
            pred = self.model(X, is_test = False)
            loss = self.loss_function(pred,y)
            
            loss.backward()
            self.optimizer.step()
            wandb.log({'Loss by batch':loss})