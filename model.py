# baseline
import torch
import torch.nn as nn
import torchaudio.transforms as ts
import torch.nn.functional as F
import arguments

sys_args, exp_args = arguments.get_args()
CPU = sys_args['cpu']
GPU = sys_args['gpu']

class Resblock(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, ps): #projection shortcut
        super(Resblock, self).__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.out_ch = out_ch
        self.ps = ps

        #self.conv1_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.hid_ch, kernel_size=(3,3), stride=2, padding=1)
        
        self.conv1 = nn.Conv2d(in_channels=self.in_ch, out_channels=self.hid_ch, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.hid_ch, out_channels=self.out_ch, kernel_size=(3,3), stride=1, padding=1)

        self.relu = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(self.hid_ch)
        self.bn2 = nn.BatchNorm2d(self.out_ch)
        
        # self.conv_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(2,2), stride=2)
        # # self.conv_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch,kernel_size=(1,1))
        # self.maxpool_ps = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.bn_ps = nn.BatchNorm2d(self.out_ch)
        self.conv_ps = nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(3,3), stride=1, padding=1)
        self.bn_ps = nn.BatchNorm2d(self.out_ch)
        
    def forward(self, x):
        # if self.ps:
        #     a = self.conv_ps(x)
        #     # a = self.maxpool_ps(a)
        # else:
        #     a = self.conv1(x)
        
        a = self.conv1(x)    
        a = self.bn1(a)
        a = self.relu(a)
        a = self.conv2(a)
        a = self.bn2(a)

        # if self.ps:
        #     x = self.conv_ps(x)
        #     # x = self.maxpool_ps(x)
        #     x = self.bn_ps(x)
        if self.ps:
            x = self.conv_ps(x)
            x = self.bn_ps(x)
            
        x = x + a

        x = self.relu(x)

        return x


class ResNet_18(nn.Module): 
    def __init__(self, embedding_size=exp_args['embedding_size']): #embedding_size -> hyperparameter 설정
        super(ResNet_18, self).__init__()
        self.melspec = ts.MelSpectrogram(
            sample_rate = exp_args['sample_rate'], 
            n_fft = exp_args['n_fft'], 
            n_mels = exp_args['n_mels'], 
            win_length = exp_args['win_length'], 
            hop_length = exp_args['hop_length'], 
            window_fn=torch.hamming_window).to(GPU)
        
        self.embedding_size = embedding_size
        
        self.preemphasis = AudioPreEmphasis(0.97)
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7), stride=2, padding=3) 
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            Resblock(32, 32, 32, False),
            Resblock(32, 32, 32, False)
        )
        self.layer2 = nn.Sequential(
            Resblock(32, 64, 64, True),
            Resblock(64, 64, 64, False),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size= 3, stride = 2, groups = 64),
            nn.BatchNorm2d(64),
            self.relu
        )
        self.layer3 = nn.Sequential(
            Resblock(64, 128, 128, True),
            Resblock(128, 128, 128,False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size= 3, stride = 2, groups = 128),
            nn.BatchNorm2d(128),
            self.relu
        )
        self.layer4 = nn.Sequential(
            Resblock(128, 256, 256, True),
            Resblock(256, 256, 256, False),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size= 3, stride = 2, groups = 256),
            nn.BatchNorm2d(256),
            self.relu
        )
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=7, stride=2, padding=3), # (:, 512, 2, 10)
        #     nn.BatchNorm2d(512),
        #     self.relu,
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1), # (:, 512, 1, 5)
        #     nn.BatchNorm2d(512),
        #     self.relu,
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        
        self.fc1 = nn.Linear(in_features=1024, out_features=self.embedding_size)
        self.bn2 = nn.BatchNorm1d(self.embedding_size)
        
        
        self.fc2 = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size)
        self.bn3 = nn.BatchNorm1d(self.embedding_size)
        #self.leaky_relu = nn.LeakyReLU()
        
        self.fc3 = nn.Linear(in_features=self.embedding_size, out_features=1211)
        self.bn4 = nn.BatchNorm1d(1211)
        
    def forward(self, x, is_test = False): # x.size = (32, 1, 4*16000)
        x = x.to(GPU)
        x = self.preemphasis(x)
        x = self.melspec(x) # (32, 1, 64, 320)
        x = torch.log(x+1e-5)        
            
        x = self.conv0(x) # (:, 32, 32, 160)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (:, 32, 32, 160)
        
        x = self.layer1(x) # (:, 32, 32, 160)
        x = self.layer2(x) # (:, 64, 16, 80)
        x = self.layer3(x) # (:, 128, 8, 40)
        x = self.layer4(x) # (:, 256, 4, 20)
        #print(x.shape)
        x = self.avgpool(x) # (:, 256, 1, 1)
        
        x = x.view(x.size(0), -1) # 

        x = self.relu(self.bn2(self.fc1(x)))
        x = self.bn3(self.fc2(x)) #        
        # x = F.normalize(x, dim = 1, p=2.)
        
        if is_test: # embedding 출력
            return x
        
        x = self.bn4(self.fc3(x))

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