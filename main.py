from mydataset import download_VOCdvkit,Mydataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from matplotlib import pyplot as plt
import torch
from utils import show_img
import os
from unet import U_Net
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#download_VOCdvkit()
data = Mydataset('VOC2012')
print(f'data{len(data)}')
#show_img(*data[0])
dataloader = DataLoader(data,batch_size=16,shuffle=True)
weight_path = 'params/unet.pth'
save_path = 'train_img'
model = U_Net().to(device)
opt = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fn = torch.nn.BCELoss()
if os.path.exists(weight_path):
  model.load_state_dict(torch.load(weight_path))
  print('load weight')
epoch = 1
for i,(x,y) in enumerate(dataloader):
    x,y = x.to(device),y.to(device)
    out = model(x)
    loss = loss_fn(out,y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f'epoch{epoch},step{i},loss{loss.item()}')
    if i%10==0:
       torch.save(model.state_dict(),weight_path)
    if i%10==0:
       _x = x[0]
       _y = y[0]
       _out = out[0]
       _img = torch.stack([_x,_y,_out],dim=0)
       save_image(_img,f'{save_path}/{i}.png')
    epoch+=1