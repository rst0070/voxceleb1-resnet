import torch
import torch.nn as nn
import torchaudio.transforms as ts
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

        self.conv1_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.hid_ch, kernel_size=(3,3), stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=self.in_ch, out_channels=self.hid_ch, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.hid_ch, out_channels=self.out_ch, kernel_size=(3,3), stride=1, padding=1)

        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(self.hid_ch)
        self.bn2 = nn.BatchNorm2d(self.out_ch)
        
        self.conv_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(2,2), stride=2)
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
            
        # print('x', x.size())
        # print('a', a.size())
        x = x + a
        x = self.bn_ps(x)
        x = self.relu(x)

        return x


class ResNet_18(nn.Module): 
    def __init__(self, embedding_size=512): #embedding_size -> hyperparameter 설정
        super(ResNet_18, self).__init__()
        self.melspec = ts.MelSpectrogram(
            sample_rate = exp_args['sample_rate'], 
            n_fft = exp_args['n_fft'], 
            n_mels = exp_args['n_mels'], 
            win_length = exp_args['win_length'], 
            hop_length = exp_args['hop_length'],
            f_min = exp_args['f_min'],
            f_max = exp_args['f_max'], 
            window_fn = torch.hamming_window
            ).to(GPU)
        
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
        
        # self.fc1 = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=512),
        #     nn.Dropout(0.25),
        #     nn.BatchNorm1d(512),
        #     self.relu
        # )
        
        # self.fc2 = nn.Sequential(
        #     nn.Linear(in_features=512, out_features=512),
        #     nn.Dropout(0.25),
        #     nn.BatchNorm1d(512),
        #     self.relu
        # )
        
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=self.embedding_size),
            nn.Dropout(0.25),
            nn.BatchNorm1d(self.embedding_size),
            self.relu
        )
         
        self.fc4 = nn.Linear(in_features=self.embedding_size, out_features=1211)
        
    def forward(self, x, is_test = False): # x.size = (32, 1, 4*16000)
        x = x.to(GPU)
        x = self.melspec(x) # (32, 1, 64, 320)
        #print(x.shape)
        x = torch.log(x+1e-5)
        # if x.size(0) == 1:
        #     x = torch.unsqueeze(x, 0)
        # print(x.size())
        x = self.conv0(x) # (32, 64, 32, 160)
        #print(x.size())

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) #(32, 64, 16, 80)
        # print(x.size())


        x = self.layer1(x) # (32, 64, 8, 40)
        # print(x.size())
        x = self.layer2(x) # (32, 128, 4, 20)
        # print(x.size())
        x = self.layer3(x) # (32, 256, 2, 10)
        # print(x.size())
        x = self.layer4(x) # (32, 512, 1, 5)
        # print(x.size())
        x = self.avgpool(x) # (32, 512, 1, 1)
        # print(x.size())
        x = x.view(x.size(0), -1) # (32, 512)
        # print(x.size())
        
        # x = self.fc1(x) # (32, 256)
        
        # x = self.fc2(x)
        
        x = self.fc3(x)
        
        if is_test: # embedding 출력
            return x
        
        x = self.fc4(x) # prediction 출력 (32, 1211)
        # print(x.size())

        return x