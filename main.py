import arguments
from model import ResNet_18
import torch
import dataset
import trainer
import tester
from tqdm import trange

class Main:
    def __init__(self):
        sys_args, exp_args = arguments.get_args()
        
        GPU = sys_args['gpu']
        
        self.max_epoch = exp_args['epoch']
        
        self.model = ResNet_18(embedding_size=exp_args['embedding_size']).to(GPU)
        
        # optimizer 정의
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = 0.001,
            weight_decay=1e-5
        )
        
        # learning rate가 epoch마다 0.95%씩 감소하도록 설정
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.95
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
            

if __name__ == '__main__':
    #torch.multiprocessing.set_start_method("spawn")
    program = Main()
    program.start()
    