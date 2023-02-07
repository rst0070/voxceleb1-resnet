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
        
        self.conv_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(1,1), stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=1)
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
            #print('before ps', x.shape)
            x = self.conv_ps(x)
            #print('after ps', x.shape)
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
            n_mels=64, 
            win_length=400, 
            hop_length=160, 
            window_fn=torch.hamming_window).to(GPU)
        
        self.embedding_size = embedding_size
        
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=2, padding=1) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.res2 = nn.Sequential(
            Resblock(in_ch = 64, hid_ch = 64, out_ch = 64, ps = False),
            Resblock(in_ch = 64, hid_ch = 64, out_ch = 64, ps = False)
        )
        
        self.res3 = nn.Sequential(
            Resblock(in_ch = 64, hid_ch = 128, out_ch = 128, ps = True),
            Resblock(in_ch = 128, hid_ch = 128, out_ch = 128, ps = False)
        )
        
        self.res4 = nn.Sequential(
            Resblock(in_ch = 128, hid_ch = 256, out_ch = 256, ps = True),
            Resblock(in_ch = 256, hid_ch = 256, out_ch = 256, ps = False)
        )
        
        self.res5 = nn.Sequential(
            Resblock(in_ch = 256, hid_ch = 512, out_ch = 512, ps = True),
            Resblock(in_ch = 512, hid_ch = 512, out_ch = 512, ps = False)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(in_features=(512), out_features=self.embedding_size) 
        self.fc2 = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size) 
        self.fc3 = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size) 
        self.fc4 = nn.Linear(in_features=self.embedding_size, out_features=1211)
        
    def forward(self, x, is_test = False): # x.size = (32, 1, 4*16000)
        x = x.to(GPU)
        x = self.melspec(x) # (batch, 1, 64, 300)
        x = x[:, :, :, 0 : 300]
        print('start', x.shape)
        x = self.conv0(x) # (batch, 64, 32, 200)
        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x) #(batch, 64, 10, 100)
        print(x.shape)

        x = self.res2(x) # (batch, 64, 10, 100)
        print('output of res2', x.shape)
        x = self.res3(x) # (32, 128, 5, 50)
        print('output of res3', x.shape)
        x = self.res4(x) # (32, 256, 3, 25)
        print('output of res4', x.shape)
        x = self.res5(x) # (32, 512, 2, 13)
        print('output of res5', x.shape)
        x = self.avgpool(x) # (32, 512, 1, 1)
        
        x = x.view(x.size(0), -1) # (32, 512)

        
        x = self.fc1(x) # (32, 256)
        x = self.fc2(x)
        x = self.fc3(x)
        
        if is_test: # embedding 출력
            return x
        
        x = self.fc4(x) # prediction 출력 (32, 1211)


        return x