import torch
from torch.utils.data import Dataset
import os
import torchaudio
import pandas as pd
from tqdm import trange

NUM_TRAIN_SPEAKER = 1211
NUM_FRAME_PER_INPUT = 16000 * 4
NUM_SEG_PER_UTTER = 28

#def resizeWaveform(waveform:torch.Tensor):
    
    

class TrainDataset(Dataset):
    
    def __init__(self, annotation_file_path, data_dir):
        
        self.annotation_table = pd.read_csv(annotation_file_path, delim_whitespace = True)
        self.num_utter = len(self.annotation_table)
        self.cache = []

        for r_idx in trange(self.num_utter, desc="loading train data"):
            path = os.path.join(data_dir, self.annotation_table.iloc[r_idx, 2])
            
            wf, _ = torchaudio.load(path)
            wf = resizeWaveform(wf)
            
            speaker_num = int(self.annotation_table.iloc[r_idx, 0])            
            self.cache.append((wf, speaker_num))
        
        
    def __len__(self):
        return self.num_utter
    
    def __getitem__(self, idx):
        return self.cache[idx]
    
class TestDataset(Dataset):
    """_summary_
    직접 발성에 대한 특징을 주는게 아니라 각 오디오파일의 id를 넘겨준다. 
    
    """
    
    def __init__(self, annotations_file_path, audio_dir):
        super().__init__()
        self.annotation_table = pd.read_csv(annotations_file_path, delim_whitespace=True)
        self.num_label = len(self.labels)
        self.id_ans_list = []
        self.all_feature = {}

        for r_idx in trange(self.num_label, desc="loading test data"):
            """
            test audio file을 다 불러오고 저장한다.
            """
            label = int(self.labels.iloc[r_idx, 0])
            # 오디오에 대한 id = path
            id1 = self.labels.iloc[r_idx, 1]
            id2 = self.labels.iloc[r_idx, 2]
            
            if id1 not in self.all_feature:
                path = audio_dir + '/' + id1
                wf, _ = torchaudio.load(path)
                self.all_feature[id1] = self.getSplittedWaveform(wf)
                
            if id2 not in self.all_feature:
                path = audio_dir + '/' + id2
                wf, _ = torchaudio.load(path)
                self.all_feature[id2] = self.getSplittedWaveform(wf)
            
            self.id_ans_list.append((id1, id2, label))
        
        print(len(self.all_feature))
        
    def getSplittedWaveform(waveform:torch.Tensor):
        waveform = resizeWaveform(waveform)
                
        _, n_frames_wf = waveform.shape
    
        start_frs = torch.linspace(start = 0, end = n_frames_wf - NUM_FRAME_PER_INPUT, steps = NUM_SEG_PER_UTTER, dtype = int)
    
        tensor_list = []
        for start in start_frs:
            end = start + NUM_FRAME_PER_INPUT
            tensor_list.append(waveform[:, start : end])
        
        return torch.concat(tensor_list, dim = 0)
            
    def __len__(self):
        return self.num_label
    
    def __getitem__(self, idx:int):
        """
        """
        return self.cache[idx]
    
    def getAllFeature(self):
        return self.all_feature