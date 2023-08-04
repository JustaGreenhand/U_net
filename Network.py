## 1.0.1
import torch.nn as nn
import torch
from torch.nn import functional as F


#conv
class Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.conv_layer = nn.Sequential(
            #one
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            #two
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),

            nn.Dropout(0.4),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.conv_layer(x)
    

#Down  
class Down(nn.Module):
    def __init__(self, in_channel):
        super(Down, self).__init__()
        self.Down_layer = nn.Sequential(
            #nn.Conv2d(in_channel, in_channel, 3,2,1),#size,strike,padding
            nn.MaxPool2d(2),
            nn.ReLU()
        )
    def forward(self, x):
        return self.Down_layer(x)
#Up
class Up(nn.Module):
    def __init__(self, in_channel):
        super(Up, self).__init__()
        # self.Up_layer = nn.Conv2d(in_channel, in_channel//2, 1,1)
        self.Up_layer = nn.ConvTranspose2d(in_channel, in_channel//2,2,2)

    def forward(self, x, res):
        # up = F.interpolate(x, scale_factor=2, mode="nearest")
        # x = self.Up_layer(up)
        # print(x.shape)
        x = self.Up_layer(x)
        # print(res.shape, x.shape)
        return torch.cat((x, res), dim=1) #拼接


class Unet(nn.Module):
    def __init__(self, in_channel : int, out_channel : int):
        super(Unet, self).__init__()
        #Down
        self.Down_Conv1 = Conv(in_channel, 64)
        self.Down1 = Down(64)

        self.Down_Conv2 = Conv(64,128)
        self.Down2 = Down(128)

        self.Down_Conv3 = Conv(128,256)
        self.Down3 = Down(256)

        self.Down_Conv4 = Conv(256,512)
        self.Down4 = Down(512)

        self.Conv = Conv(512,1024)

        #Up
        self.Up1 = Up(1024)
        self.Up_Conv1 = Conv(1024,512)

        self.Up2 = Up(512)
        self.Up_Conv2 = Conv(512,256)
        
        self.Up3 = Up(256)
        self.Up_Conv3 = Conv(256,128)

        self.Up4 = Up(128)
        self.Up_Conv4 = Conv(128,64)

        self.pred = nn.Conv2d(64,out_channel,3,1,1)#in out size strike padding

    def forward(self, x):
        #Down
        D1 = self.Down_Conv1(x)
        D2 = self.Down_Conv2(self.Down1(D1))
        D3 = self.Down_Conv3(self.Down2(D2))
        D4 = self.Down_Conv4(self.Down3(D3))

        Y = self.Conv(self.Down4(D4))
        # print(Y.shape, D4.shape)
        
        U1 = self.Up_Conv1(self.Up1(Y, D4))
        U2 = self.Up_Conv2(self.Up2(U1, D3))
        U3 = self.Up_Conv3(self.Up3(U2, D2))
        U4 = self.Up_Conv4(self.Up4(U3, D1))

        return F.sigmoid(self.pred(U4))
        # return self.pred(U4)
    

# if __name__ == '__main__':
#     a = torch.randn(2,3,256,256)
#     net = Unet()
#     print(net(a).shape)