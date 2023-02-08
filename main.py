import arguments
from model import ResNet_18
import torch
import dataset
import trainer
import tester
from tqdm import trange
import os
import wandb

class Main:
    def __init__(self):
        sys_args, exp_args = arguments.get_args()
        
        GPU = sys_args['gpu']
        self.lr = exp_args['lr']
        self.embedding_size = exp_args['embedding_size']
        self.max_epoch = exp_args['epoch']
        self.batch_size = exp_args['batch_size']

        os.system('wandb login be65d6ddace6bf4e2441a82af03c144eb85bbe65')
        wandb.init(project='resnet18-fc4-preemphasis-0.97', entity='irlab_undgrd')
        wandb.config = {
            "learning_rate" : self.lr,
            "epochs" : self.max_epoch,
            "batch_size" : self.batch_size
        }
        wandb.define_metric("loss")
        wandb.define_metric("eer")

        
        self.model = ResNet_18(embedding_size=exp_args['embedding_size']).to(GPU)
        
        # optimizer 정의
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = exp_args['lr'],
            weight_decay = exp_args['weight_decay'],
            amsgrad=True,
        )
        
        # learning rate가 epoch마다 0.95%씩 감소하도록 설정
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.97
        )
        
        train_dataset = dataset.TrainDataset(annotation_file_path = sys_args['path_train_label'], data_dir = sys_args['path_train'])
        test_dataset = dataset.TestDataset(annotation_file_path = sys_args['path_test_label'], data_dir = sys_args['path_test'])
        
        self.trainer = trainer.Trainer(model = self.model, dataset = train_dataset, optimizer = optimizer, batch_size = exp_args['batch_size'])
        self.tester = tester.Tester(model = self.model, dataset = test_dataset, batch_size=exp_args['batch_size'])
        
    def start(self):
        
        for epoch in range(1, self.max_epoch + 1):
            
            self.trainer.train()
            
            self.tester.test(epoch = epoch)
            
            self.lr_scheduler.step()
            
    def save(self):
        sys_args, exp_args = arguments.get_args()
        torch.save(self.model, sys_args['path_save'])
            

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method("spawn")
    program = Main()
    program.start()
    