# tester.py
from model import ResNet_18
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import TestDataset
import torch.nn.functional as F

import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import wandb

from arguments import get_args

sys_args, exp_args = get_args()

EMBEDDING_SIZE = exp_args['embedding_size']
SAMPLE_NUM = exp_args['test_sample_num']
NUM_FRAMES = exp_args['num_train_frames']

GPU = sys_args['gpu']
CPU = sys_args['cpu']

class Tester:
    
    def __init__(self, model:ResNet_18, dataset:TestDataset, batch_size):
        
        self.model = model
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
    def prepareEmbedding(self):
        data_dict = self.dataset.getAllFeature()
        self.embs = {}
        
        with torch.no_grad():
            for audio_id, features in tqdm(data_dict.items(), desc="getting embeddings"):
                
                _, n_fr = features.shape # frame 개수
                
                fragments = torch.linspace(start = 0, end = n_fr - NUM_FRAMES, steps = SAMPLE_NUM, dtype = int) # 5개 구간 추출. 각 구간이 시작하는 지점을 저장한다.
                
                feature_list = []
                
                for fragment in fragments: # 각 구간의 시작점에서 부터 frames 만큼 잘라서 사용한다.
                    feature_list.append(features[:, fragment:fragment + int(NUM_FRAMES)])
                
                feature_list = torch.stack(feature_list, dim = 0).to(GPU)
                  
                embeddings = self.model(feature_list, is_test = True)
                
                self.embs[audio_id] = torch.sum(embeddings, dim = 0) / SAMPLE_NUM
            
            
    def idListToEmbListTensor(self, id_list):
        """
        [
            [ embedding 1],
            [ embedding 2],
            ...
        ]
        """
        result = []
        
        
        
        for audio_id in id_list:
            result.append(self.embs[audio_id])
            
        return torch.stack(result, dim=0)
    
        
        
    def getEER(self, labels, cos_sims):
        labels = labels.to(CPU)
        cos_sims = cos_sims.to(CPU)
        fpr, tpr, _ = metrics.roc_curve(labels, cos_sims, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer
    
    def test(self, epoch):
        
        self.model.eval()
        
        self.prepareEmbedding()
        
        sims = []
        labels = []
        
        for audio_id1, audio_id2, label in tqdm(self.loader, desc="testing"):
            
            embs1 = self.idListToEmbListTensor(audio_id1).to(GPU) # 2차원 [id, node_idx]
            embs2 = self.idListToEmbListTensor(audio_id2).to(GPU)
            
            #print('*******', embs1.shape)
            
            sim = F.cosine_similarity(embs1, embs2, dim = 1).to(CPU) # 1차원 형태
            #print(sim)
            sims.append(sim)
            labels.append(label)
        sims = torch.cat(sims, dim = 0)
        labels = torch.cat(labels, dim = 0)
        eer = self.getEER(labels, sims)
        print(f"epoch: {epoch}, EER: {eer}")
        # wandb.log({"EER by epoch" : eer})