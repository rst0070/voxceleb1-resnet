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
        self.loss_function = nn.CrossEntropyLoss().to(GPU)
        self.optimizer = optimizer

    def train(self):
        self.model.train()
        
        itered = 0
        loss_sum = 0
        
        pbar = tqdm(self.dataloader)
        for X, y in pbar:
            itered = itered + 1
            self.optimizer.zero_grad()
            
            X = X.to(GPU)
            y = y.to(GPU)
            
            pred = self.model(X, is_test = False)
            loss = self.loss_function(pred,y)
            pbar.set_description(f"training loss: {loss}")
            
            loss_sum += loss
            
            loss.backward()
            self.optimizer.step()
            
            if itered == 50:
                wandb.log({'Loss':loss_sum / float(itered)})
                itered = 0
                loss_sum = 0
                
        mean =  loss_sum / float(itered)       
        wandb.log({'Loss': mean})
        print(f"loss mean: {mean}")        
                