# baseline
import torch
import torch.nn as nn
import torchaudio.transforms as ts
import torch.nn.functional as F
import arguments

sys_args, exp_args = arguments.get_args()
GPU = sys_args['gpu']

class Resblock(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, ps): #projection shortcut
        super(Resblock, self).__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.out_ch = out_ch
        self.ps = ps

        self.conv1_ps = nn.Conv1d(in_channels=self.in_ch, out_channels=self.hid_ch, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv1d(in_channels=self.in_ch, out_channels=self.hid_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.hid_ch, out_channels=self.out_ch, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm1d(self.hid_ch)
        self.bn2 = nn.BatchNorm1d(self.out_ch)
        
        self.conv_ps = nn.Conv1d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=2, stride=2)
        self.bn_ps = nn.BatchNorm1d(self.out_ch)

    def forward(self, x):
        if self.ps:
            a = self.conv_ps(x)
        else:
            a = self.conv1(x)
            
        a = self.bn1(a)
        a = self.relu(a)
        a = self.conv2(a)
        a = self.bn2(a)

        if self.ps:
            x = self.conv_ps(x)
            x = self.bn_ps(x)
        x = x + a

        x = self.relu(x)

        return x


class ResNet_18(nn.Module): 
    def __init__(self, embedding_size=exp_args['embedding_size']): #embedding_size -> hyperparameter 설정
        super(ResNet_18, self).__init__()
        
        self.embedding_size = embedding_size
        
        self.preemphasis = AudioPreEmphasis(0.97)
        self.conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3) 
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(self.embedding_size)
        self.bn5 = nn.BatchNorm1d(1211)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

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
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_features=512, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=self.embedding_size) 
        self.fc4 = nn.Linear(in_features=self.embedding_size, out_features=1211)
        
    def forward(self, x, is_test = False): # x.size = (32, 1, 4*16000)
        x = x.to(GPU)
        x = self.preemphasis(x)
        x = F.normalize(x, p = 2., dim = 2) # [:, 1, 51200]
        
        x = self.conv0(x) #  [-1, 64, 25600]

        x = self.bn1(x) # [-1, 64, 25600]
        x = self.relu(x) # [-1, 64, 25600]
        x = self.maxpool(x) # [-1, 64, 12800]
        x = self.layer1(x) # [-1, 64, 12800]
        x = self.layer2(x) #  [-1, 128, 6400] 
        x = self.layer3(x) # [-1, 256, 3200]
        x = self.layer4(x) # [-1, 512, 1600]

        x = self.avgpool(x) # [-1, 512, 1]
        
        x = x.view(x.size(0), -1) # 

        
        x = self.relu(self.bn2(self.fc1(x))) #
        x = self.relu(self.bn3(self.fc2(x)))
        x = self.bn4(self.fc3(x))
        
        if is_test: # embedding 출력
            return x
        
        x = self.bn5(self.fc4(x))

        return x

class AudioPreEmphasis(nn.Module):

    def __init__(self, coeff=0.97):
        super().__init__()

        self.w = torch.FloatTensor([-coeff, 1.0]).unsqueeze(0).unsqueeze(0)

    def forward(self, audio):
        audio = F.pad(audio,(1,0), 'reflect')
        return F.conv1d(audio, self.w.to(audio.device))

if __name__ == '__main__':
    from torchsummary import summary
    
    model = ResNet_18().cuda()
    summary(model, input_size=(1,int(16000*3.2)))