import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as FT
import numpy as np
from PIL import Image
from dataset import PneumoDataset

class Pad():
    def __init__(self, max_value):
        self.max_value = max_value
        
    def __call__(self, sample):
        image, mask = sample
        pad_val = np.random.randint(0, self.max_value)
        image = FT.pad(image, padding=pad_val, fill=0)
        mask = FT.pad(mask, padding=pad_val, fill=0)
        return image, mask


class Crop():
    def __init__(self, max_val):
        self.max_val = max_val
        
    def __call__(self, sample):
        image, mask = sample
        tl_shift = np.random.randint(0, self.max_val)
        br_shift = np.random.randint(0, self.max_val)
        im_w, im_h = image.size
        crop_w = im_w - tl_shift - br_shift
        crop_h = im_h - tl_shift - br_shift
        
        image = FT.crop(image, tl_shift, tl_shift,crop_h, crop_w)
        mask = FT.crop(mask, tl_shift, tl_shift,crop_h, crop_w)
        return image, mask


class Resize():
    def __init__(self, output_size):
        self.output_size = output_size
        
    def __call__(self, sample):
        image, mask = sample
        image = FT.resize(image, self.output_size)
        mask = FT.resize(mask, self.output_size)
        
        return image, mask

# Plotting functions
def blend(img_tensor, mask1=None, mask2=None):
    image = FT.to_pil_image(img_tensor + 0.5).convert("RGB")
    if mask1 is not None:
        mask1 =  FT.to_pil_image(torch.cat([
            torch.zeros_like(img_tensor),
            torch.stack([mask1.float()]),
            torch.zeros_like(img_tensor)
        ]))
        image = Image.blend(image, mask1, 0.2)
        
    if mask2 is not None:
        mask2 =  FT.to_pil_image(torch.cat([
            torch.stack([mask2.float()]),
            torch.zeros_like(img_tensor),
            torch.zeros_like(img_tensor)
        ]))
        image = Image.blend(image, mask2, 0.2)
    
    return np.array(image)



# Model functions
def save_checkpoint(state, filename= "/content/pneumothorax_checkpoint.pth.tar"):
  print(">>Saving Model")
  torch.save(state, filename)

def load_checkpoint(checkpoint, model):
  print(">>Loading Model")
  model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_img_dir, train_mask_dir, valid_img_dir, valid_mask_dir,
                batch_size, train_transforms, valid_transforms, pin_memory, num_workers):
  
  # creating train dataset
  train_dataset= PneumoDataset(image_dir= train_img_dir,
                               mask_dir= train_mask_dir,
                               transforms= train_transforms,shuffle= True)
  
  # creating validation dataset from paths
  valid_dataset= PneumoDataset(image_dir= valid_img_dir,
                               mask_dir= valid_mask_dir,
                               transforms= valid_transforms, shuffle= True)
  
  # Dataloader for train and validation
  train_loader= DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True, pin_memory= pin_memory,
                           num_workers= num_workers)
  
  valid_loader= DataLoader(dataset= valid_dataset, batch_size= batch_size,pin_memory= pin_memory,
                           num_workers= num_workers)

  return train_loader, valid_loader
