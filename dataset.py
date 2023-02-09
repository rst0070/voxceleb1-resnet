import torch
from torch.utils.data import Dataset
import os
import torchaudio
import pandas as pd
from tqdm import trange
import arguments
import random

_, exp_args = arguments.get_args()

#NUM_TRAIN_SPEAKER = 1211
NUM_FRAME_PER_INPUT = int(exp_args['num_train_frames'])

def resizeWaveform(waveform:torch.Tensor):
    """_summary_
    waveform의 frame수는 NUM_FRAME_PER_INPUT 이상이어야한다.
    """
    _, n_frames_wf = waveform.shape
    
    # 길이조정 필요없는경우
    if n_frames_wf >= NUM_FRAME_PER_INPUT:
        return waveform
        
    residue = NUM_FRAME_PER_INPUT % n_frames_wf
    tensor_list = []
            
    for i in range(0, NUM_FRAME_PER_INPUT // n_frames_wf):
        tensor_list.append(waveform)
    if residue > 0:
        tensor_list.append(waveform[:, 0:residue])
                
    return torch.cat(tensor_list, 1)

    

class TrainDataset(Dataset):
    
    def __init__(self, annotation_file_path, data_dir):
        super().__init__()
        self.annotation_table = pd.read_csv(annotation_file_path, delim_whitespace = True)
        self.num_utter = len(self.annotation_table)
        self.data_dir = data_dir
        
    def __len__(self):
        return self.num_utter
    
    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.annotation_table.iloc[idx, 2])
        wf, _ = torchaudio.load(path)
        speaker_num = int(self.annotation_table.iloc[idx, 0]) - 1 # 0부터 1210까지 번호가 매겨진 화자들
        
        wf = resizeWaveform(wf)
        
        _, n_fr = wf.shape
        
        start_fr = random.randint(0, n_fr - NUM_FRAME_PER_INPUT)
        return wf[:, start_fr : start_fr + NUM_FRAME_PER_INPUT], speaker_num
    
class TestDataset(Dataset):
    """_summary_
    직접 발성에 대한 특징을 주는게 아니라 각 오디오파일의 id를 넘겨준다. 
    
    """
    
    def __init__(self, annotation_file_path, data_dir):
        super().__init__()
        self.annotation_table = pd.read_csv(annotation_file_path, delim_whitespace=True)
        self.num_label = len(self.annotation_table)
        self.id_to_waveform = {} # 오디오파일의 id와 waveform 대응
        self.cache = [] # getitem 

        for r_idx in trange(self.num_label, desc="loading test data"):
            """
            test audio file을 다 불러오고 저장한다.
            """
            label = int(self.annotation_table.iloc[r_idx, 0])
            # 오디오에 대한 id = path
            id1 = self.annotation_table.iloc[r_idx, 1]
            id2 = self.annotation_table.iloc[r_idx, 2]
                        
            if id1 not in self.id_to_waveform:   
                path = data_dir + '/' + id1
                wf, _ = torchaudio.load(path)
                wf = resizeWaveform(wf)
                self.id_to_waveform[id1] = wf                
                
            if id2 not in self.id_to_waveform:
                path = data_dir + '/' + id2
                wf, _ = torchaudio.load(path)
                wf = resizeWaveform(wf)
                self.id_to_waveform[id2] = wf  
            
            self.cache.append((id1, id2, label))
            
    def __len__(self):
        return self.num_label
    
    def __getitem__(self, idx:int):
        """
        """
        return self.cache[idx]
    
    def getAllFeature(self):
        return self.id_to_waveform