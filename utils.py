from PIL import Image
import torch
from matplotlib import pyplot as plt
def keep_image_size(path=str,size=(256,256)):
    image = Image.open(path).convert('RGB')
    temp = max(image.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(image,(0,0))
    mask=mask.resize(size)
    return mask

def show_img(A:torch.Tensor,B:torch.Tensor):
  plt.subplot(1,2,1)
  plt.imshow(A.permute(1,2,0))
  plt.subplot(1,2,2) 
  plt.imshow(B.permute(1,2,0))
  plt.show()