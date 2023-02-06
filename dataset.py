import torch
from torch.utils.data import Dataset
import os
import torchaudio
import pandas as pd
from tqdm import trange

NUM_TRAIN_SPEAKER = 1211
NUM_FRAME_PER_INPUT = 16000 * 4

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
        self.id_to_waveform = {}
        self.cache = []

        for r_idx in trange(self.num_label, desc="loading test data"):
            """
            test audio file을 다 불러오고 저장한다.
            """
            label = int(self.labels.iloc[r_idx, 0])
            # 오디오에 대한 id = path
            id1 = self.labels.iloc[r_idx, 1]
            id2 = self.labels.iloc[r_idx, 2]
                        
            if id1 not in self.id_to_waveform:   
                path = audio_dir + '/' + id1
                wf, _ = torchaudio.load(path)
                wf = resizeWaveform(wf)
                self.id_to_waveform[id1] = wf                
                
            if id2 not in self.all_feature:
                path = audio_dir + '/' + id2
                wf, _ = torchaudio.load(path)
                wf = resizeWaveform(wf)
                self.id_to_waveform[id2] = wf  
            
            self.cache.append((self.id_to_waveform[id1], id1, self.id_to_waveform[id2], id2, label))
            
    def __len__(self):
        return self.num_label
    
    def __getitem__(self, idx:int):
        """
        `(waveform1:torch.Tensor, wf_id1:str, waveform2:torch.Tensor, wf_id2:str, label:int)`을 return 한다. 
        - `waveform1` - 화자1의 특정 발성에 대한 waveform
            - 이것의 shape은 `[1, sec * 16000]`이다. 이때 `sec`은 4이상    
  
        - `wf_id1` - `waveform1`에 대한 고유한 id  
  
        - `waveform2` - 화자2의 특정 발성에 대한 waveform
            - 이것의 shape은 `[1, sec * 16000]`이다. 이때 `sec`은 4이상  
  
        - `wf_id2` - `waveform2`에 대한 고유한 id  
  
        - `label` - 화자1과 화자2가 동일인물인지 나타내는 라벨.
            - `0` - 다른 화자
            - `1` - 같은 화자
        """
        return self.cache[idx]