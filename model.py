# baseline
import torch
import torch.nn as nn
import torchaudio.transforms as ts
import torch.nn.functional as F
import arguments

sys_args, exp_args = arguments.get_args()
CPU = sys_args['cpu']
GPU = sys_args['gpu']

class AudioPreEmphasis(nn.Module):

    def __init__(self, coeff=0.97):
        super().__init__()

        self.w = torch.FloatTensor([-coeff, 1.0]).unsqueeze(0).unsqueeze(0)

    def forward(self, audio):
        audio = F.pad(audio,(1,0), 'reflect')
        return F.conv1d(audio, self.w.to(audio.device))
    
class FeatureExtract(nn.Module):
    """
    log mel spec과 waveform에서 해당 window의 중요도(커널과 waveform의 유사도)를 곱하는 연산
    32개의 channel로 나온다.
    """
    def __init__(self):
        super().__init__()
        self.preemphasis = AudioPreEmphasis(0.97)
        
        self.melspec = ts.MelSpectrogram(
            sample_rate = exp_args['sample_rate'], 
            n_fft = exp_args['n_fft'], 
            n_mels = exp_args['n_mels'], 
            win_length = exp_args['win_length'], 
            hop_length = exp_args['hop_length'], 
            window_fn=torch.hamming_window
        ).to(GPU)
        self.out_channels = 32
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.out_channels, kernel_size=exp_args['win_length'], stride = exp_args['hop_length'], padding = 160)
        self.bn = nn.BatchNorm1d(num_features=32)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, waveform):
        waveform = waveform.to(GPU)
        waveform = self.preemphasis(waveform)
        
        # A : mel spectrogram
        A = self.melspec(waveform) # (-1 , 1, 64, 320)
        A = torch.log(A+1e-12) 
        A = A.repeat(1, self.out_channels, 1, 1) # (-1, self.out_channels, 64, 320)
        
        # x : importance per window of mel spec
        x = self.conv(waveform)
        x = self.bn(x)
        x = self.sigmoid(x) # (-1, self.out_channels, 320)
        x = x.unsqueeze(dim = 2) # (-1, self.out_channels, 1, 320)
        
        y = A * x # 각 dim=1에 해당하는 행렬과 벡터간에 열 곱셈을 한다.
        return y 
        
        
        


class Resblock(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, ps): #projection shortcut
        super(Resblock, self).__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.out_ch = out_ch
        self.ps = ps

        self.conv1_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.hid_ch, kernel_size=(3,3), stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=self.in_ch, out_channels=self.hid_ch, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.hid_ch, out_channels=self.out_ch, kernel_size=(3,3), stride=1, padding=1)

        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(self.hid_ch)
        self.bn2 = nn.BatchNorm2d(self.out_ch)
        
        self.conv_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(2,2), stride=2)
        # self.conv_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch,kernel_size=(1,1))
        self.maxpool_ps = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn_ps = nn.BatchNorm2d(self.out_ch)

    def forward(self, x):
        if self.ps:
            a = self.conv_ps(x)
            # a = self.maxpool_ps(a)
        else:
            a = self.conv1(x)
            
        a = self.bn1(a)
        a = self.relu(a)
        a = self.conv2(a)
        a = self.bn2(a)

        if self.ps:
            x = self.conv_ps(x)
            # x = self.maxpool_ps(x)
            x = self.bn_ps(x)
        x = x + a

        x = self.relu(x)

        return x


class ResNet_18(nn.Module): 
    def __init__(self, embedding_size=exp_args['embedding_size']): #embedding_size -> hyperparameter 설정
        super(ResNet_18, self).__init__()
        
        
        self.embedding_size = embedding_size
        
        self.feature_extract = FeatureExtract()
        
        self.conv0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=3) 
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = nn.Sequential(
            Resblock(64, 64, 64, False),
            Resblock(64, 64, 64, False)
        )
        self.layer2 = nn.Sequential(
            Resblock(64, 128, 128, True),
            Resblock(128, 128, 128, False)
        )
        self.layer3 = nn.Sequential(
            Resblock(128, 256, 256, True),
            Resblock(256, 256, 256,False)
        )
        self.layer4 = nn.Sequential(
            Resblock(256, 512, 512, True),
            Resblock(512, 512, 512, False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(in_features=256, out_features=self.embedding_size)
        self.bn2 = nn.BatchNorm1d(self.embedding_size)
        
        self.fc3 =  nn.Linear(in_features=self.embedding_size, out_features=1211)
        self.bn3 = nn.BatchNorm1d(1211)
        
    def forward(self, x, is_test = False): # x.size = (32, 1, 4*16000)
        x = x.to(GPU)
        
        x = self.feature_extract(x)       
        
        x = self.conv0(x) # 
        x = self.bn0(x)
        x = self.relu(x)
        x = self.maxpool(x) #
        
        x = self.layer1(x) # 
        x = self.layer2(x) # 
        x = self.layer3(x) # 
        x = self.layer4(x) # 

        x = self.avgpool(x) # 
        
        x = x.view(x.size(0), -1) # 

        
        x = self.relu(self.bn1(self.fc1(x))) #
        x = self.bn2(self.fc2(x)) # [batch, embedding_size]
        
        if is_test: # embedding 출력
            return x
        
        x = self.bn3(self.fc3(x))

        return x



if __name__ == '__main__':
    from torchsummary import summary
    
    model = ResNet_18().cuda()
    summary(model, input_size=(1,int(16000*3.2 - 1)))
    # model = FeatureExtract().cuda()
    # summary(model, input_size=(1,int(16000*3.2 - 1)))