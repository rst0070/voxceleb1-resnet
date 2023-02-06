import torch
import torch.nn as nn
import torchaudio.transforms as ts
import arguments

sys_args, _ = arguments.get_args()
GPU = sys_args['gpu']

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
        
        self.conv_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(1,1), stride=2)
        self.bn_ps = nn.BatchNorm2d(self.out_ch)

    def forward(self, x):
        if self.ps:
            a = self.conv1_ps(x)
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
    def __init__(self, embedding_size=128): #embedding_size -> hyperparameter 설정
        super(ResNet_18, self).__init__()
        self.melspec = ts.MelSpectrogram(
            sample_rate=16000, 
            n_fft=512, 
            n_mels=40, 
            win_length=int(25*0.001*16000), 
            hop_length=int(10*0.001*16000), 
            window_fn=torch.hamming_window).to(GPU)
        
        self.embedding_size = embedding_size
        
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

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
        self.fc1 = nn.Linear(in_features=(512), out_features=self.embedding_size) 
        self.fc2 = nn.Linear(in_features=self.embedding_size, out_features=1211)
        
    def forward(self, x, is_test = False): # x.size = (32, 1, 4*16000)
        x = x.to(GPU)
        x = self.melspec(x) # (32, 1, 40, 400)

        
        x = self.conv0(x) # (32, 64, 20, 200)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) #(32, 64, 10, 100)


        x = self.layer1(x) # (32, 64, 10, 100)

        x = self.layer2(x) # (32, 128, 5, 50)

        x = self.layer3(x) # (32, 256, 3, 25)

        x = self.layer4(x) # (32, 512, 2, 13)

        
        x = self.avgpool(x) # (32, 512, 1, 1)

        
        x = x.view(x.size(0), -1) # (32, 512)

        
        x = self.fc1(x) # (32, 256)

        
        if is_test: # embedding 출력
            return x
        
        x = self.fc2(x) # prediction 출력 (32, 1211)


        return x