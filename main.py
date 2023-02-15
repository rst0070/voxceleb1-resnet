import arguments
from model import ResNet_18
import torch
import dataset
import trainer
import tester
import os
import wandb
import random
import torch.backends.cudnn as cudnn
import numpy as np
import sys

class Main:
    def __init__(self):
        self.sys_args, self.exp_args = arguments.get_args()
        
        if self.sys_args['wandb_disabled']: # arguments에 wandb 설정확인(wandb loggin 끄는 코드)
            os.system("wandb disabled")
            
        os.system(f"wandb login {self.sys_args['wandb_key']}") # arguments의 sys_args['wandb_key']에 자신의 key 입력 필요
        wandb.init(
            project = self.sys_args['wandb_project'],
            entity = self.sys_args['wandb_entity'],
            name = "multilabel-random-baseline-epoch50-v3"
        )
        
        
        GPU = self.sys_args['gpu']
        
        self.max_epoch = self.exp_args['epoch']
        
        self.model = ResNet_18(embedding_size=self.exp_args['embedding_size']).to(GPU)

        # seed 고정
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(0)
        
        # optimizer 정의
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = self.exp_args['lr'],
            weight_decay = self.exp_args['weight_decay']
        )
        
        # learning rate가 epoch마다 0.95%씩 감소하도록 설정
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.97
        )
        
        self.train_dataset = dataset.TrainDataset(annotation_file_path = self.sys_args['path_train_label'], data_dir = self.sys_args['path_train'])
        self.test_dataset = dataset.TestDataset(annotation_file_path = self.sys_args['path_test_label'], data_dir = self.sys_args['path_test'])
        
        self.trainer = trainer.Trainer(model = self.model, dataset = self.train_dataset, optimizer = optimizer, batch_size = self.exp_args['batch_size'])
        self.tester = tester.Tester(model = self.model, dataset = self.test_dataset, batch_size=self.exp_args['batch_size'])
        
    def start(self):

        best_eer = 99
        self.train_dataset.changeIntoMultiLabel(False)

        for epoch in range(1, self.max_epoch + 1):
            
            self.trainer.train(multilabel=False)
            eer = self.tester.test(epoch = epoch)
            
            self.lr_scheduler.step()

            wandb.log({'epoch': epoch})

            if eer <= best_eer:
                best_eer = eer
                wandb.log({"best eer" : best_eer*100})

        self.save(epoch,'baseline-v3')
        pre_epoch = epoch
        self.train_dataset.changeIntoMultiLabel(True)

        for epoch in range(1, self.max_epoch + 1):
            epoch = pre_epoch + epoch

            self.trainer.train(multilabel=True)
            
            eer = self.tester.test(epoch = epoch)
            
            self.lr_scheduler.step()

            wandb.log({'epoch': epoch})

            if eer <= best_eer:
                best_eer = eer
                wandb.log({"best eer" : best_eer*100})
            if epoch % 10 == 0:
                self.save(epoch,'multilabel-finetuned')

        self.save(epoch,'multilabel-finetuned')

            
    def save(self,epoch,desc):
        # sys_args, exp_args = arguments.get_args()
        torch.save(self.model, self.sys_args['path_save']+desc+str(epoch)+'.pt')
    
    def restart(self,path):
        model = torch.load(path)
        ct = 0
        for child in model.children():
            if ct < 16:
                for param in child.parameters():
                    param.requires_grad = False
        self.train_dataset.changeIntoMultiLabel(True)
        best_eer = 99
        pre_epoch = 50
        for epoch in range(1, 80 + 1):
            epoch = pre_epoch + epoch

            self.trainer.train(multilabel=True)
            
            eer = self.tester.test(epoch = epoch)
            
            self.lr_scheduler.step()

            wandb.log({'epoch': epoch})

            if eer <= best_eer:
                best_eer = eer
                wandb.log({"best eer" : best_eer*100})
            if epoch % 10 == 0:
                torch.save(model,self.sys_args['path_save']+'multilabel-'+str(epoch)+'.pt')
            

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method("spawn")
    program = Main()
    program.start()
    # program.restart('baseline50')
    