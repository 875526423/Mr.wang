import os
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

import kagglehub
def download_VOCdvkit():
# Download latest version
  path = kagglehub.dataset_download("mihhayou/vocdvkit")
  print("Path to dataset files:", path)
  data = Mydataset(path+'/VOCdevkit/VOC2012')
  print(len(data))

class Mydataset(Dataset):
  def __init__(self,path):
    self.path = path
    self.name = os.listdir(os.path.join(path,'SegmentationClass'))
  
  def __len__(self):
    return len(self.name)

  def __getitem__(self,index):
    segment_name = self.name[index] #xx.png
    segment_path = os.path.join(self.path,'SegmentationClass',segment_name)
    image_path = os.path.join(self.path,'JPEGImages',segment_name.replace('png','jpg'))#JPEGImages xx.jpg
    segment_image = keep_image_size(segment_path)
    image = keep_image_size(image_path)
    return transform(image),transform(segment_image)

