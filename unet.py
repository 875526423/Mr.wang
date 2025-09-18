from torch import nn
import torch
from torch.nn import functional as F
class Conv_Block(nn.Module):
  def __init__(self,in_channel,out_channel):
    super(Conv_Block,self).__init__()
    self.layer = nn.Sequential(
      nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias=False),
      #后面有BN则bias为false，节约计算量
      nn.BatchNorm2d(out_channel),
      nn.Dropout(0.3),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,stride=1,padding=1,padding_mode='reflect',bias=False),
      #后面有BN则bias为false，节约计算量
      nn.BatchNorm2d(out_channel),
      nn.Dropout(0.3),
      nn.LeakyReLU()
    )
  def forward(self,x):
    return self.layer(x)

class Down_Sample(nn.Module):
  def __init__(self,channel):
    super(Down_Sample,self).__init__()
    self.layer = nn.Sequential(
        nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=3,stride=2,padding=1,padding_mode='reflect',bias=False),
        nn.BatchNorm2d(channel),
        nn.LeakyReLU()
    )
    
  def forward(self,x):
    return self.layer(x)

class Up_Sample(nn.Module):
  def __init__(self,channel):
    super(Up_Sample,self).__init__()
    self.layer = nn.Conv2d(in_channels=channel,out_channels=channel//2,kernel_size=1,stride=1)
  def forward(self,x,feature_map):
    up = F.interpolate(x,scale_factor=2,mode='nearest')
    out = self.layer(up)
    return torch.cat((out,feature_map),dim=1)

class U_Net(nn.Module):
  def __init__(self):
    super(U_Net,self).__init__()
    self.c1 = Conv_Block(3,64)
    self.d1 = Down_Sample(64)
    self.c2 = Conv_Block(64,128)
    self.d2 = Down_Sample(128)
    self.c3 = Conv_Block(128,256)
    self.d3 = Down_Sample(256)
    self.c4 = Conv_Block(256,512)
    self.d4 = Down_Sample(512)
    self.c5 = Conv_Block(512,1024)
    self.u1 = Up_Sample(1024)
    self.c6 = Conv_Block(1024,512)
    self.u2 = Up_Sample(512)
    self.c7 = Conv_Block(512,256)
    self.u3 = Up_Sample(256)
    self.c8 = Conv_Block(256,128)
    self.u4 = Up_Sample(128)
    self.c9 = Conv_Block(128,64)
    self.out = nn.Conv2d(64,3,3,1,1)
    self.Th = nn.Sigmoid()

  def forward(self,x):
    R1 = self.c1(x)
    R2 = self.c2(self.d1(R1))
    R3 = self.c3(self.d2(R2))
    R4 = self.c4(self.d3(R3))
    R5 = self.c5(self.d4(R4))
    O1 = self.c6(self.u1(R5,R4))
    O2 = self.c7(self.u2(O1,R3))
    O3 = self.c8(self.u3(O2,R2))
    O4 = self.c9(self.u4(O3,R1))
    return self.Th(self.out(O4))