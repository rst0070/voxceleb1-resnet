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
embedding_size = exp_args['embedding_size']
sample_num = exp_args['sample_num']


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
        
        frames = 16000*4 # 4초 프레임 개수
        self.model.eval()
        with torch.no_grad():
            for audio_id, features in tqdm(data_dict.items(), desc="getting embeddings"):
                
                fragments = np.linspace(0,features.shape[0]-frames,sample_num) # 5개 구간 추출. 각 구간이 시작하는 지점을 저장한다.
                
                fragments = fragments.astype(np.int64)
                temp_embedding = torch.zeros(1,embedding_size)
                
                for fragment in fragments: # 각 구간의 시작점에서 부터 frames 만큼 잘라서 사용한다.
                    if features.shape[0] >= frames:
                        temp_feature = features[fragment:fragment+frames]
                    else:
                        temp_feature = np.append(features, np.zeros(frames - features.shape[0]))
                    temp_feature = torch.FloatTensor(temp_feature).to(GPU)
                    temp_feature = torch.unsqueeze(torch.unsqueeze(temp_feature,0),0)
                    temp_feature = self.model(temp_feature, is_test = True)
                    
                    temp_embedding += temp_feature.to(CPU)
                self.embs[audio_id] = temp_embedding / sample_num
            
            
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
            
            sim = F.cosine_similarity(embs1, embs2, dim = 1).to(CPU) # 1차원 형태
            #print(sim)
            sims.append(sim)
            labels.append(label)
        sims = torch.concat(sims, dim = 0)
        labels = torch.concat(labels, dim = 0)
        eer = self.getEER(labels, sims)
        print(f"epoch: {epoch}, EER: {eer}")
        wandb.log({"EER by epoch" : eer})