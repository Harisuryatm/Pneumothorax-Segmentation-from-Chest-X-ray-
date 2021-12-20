# importing library
import os
import numpy as np
import random
import shutil
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as FT
from torch.utils.data import Dataset


def make_dirs():
  """
  making the directory and subdirectories for training and validation images and masks
  parameters: None
  return    : None
  """
  # creating a new folder Dataset
  os.mkdir(path= "/content/Dataset")
  root_dir= "/content/Dataset"
  # 
  folders= ["train_images","train_masks","valid_images","valid_masks"]

  for fname in folders:
    os.mkdir(os.path.join(root_dir, fname))
  
  print("Folders for both training and validation are created")



def train_valid_split(dataset, source_dir, dest_dir):
  """
  Moves images from source directory to the destination directory 
  parameters -  dataset   :  csv file containing the image details
                source_dir:  directory from source
                dest_dir  :  directory to move images to destination  
  """
  # getting equal number of pneumothroax and non-pneumothorax patients
  pneumo_files= list(dataset[dataset["has_pneumo"] == 1]["new_filename"].values)
  n_pneumo= len(pneumo_files)
  nonpneumo_files= list(dataset[dataset["has_pneumo"] == 0].head(n_pneumo)["new_filename"].values)
  
  # files to be stored
  file_names= pneumo_files+nonpneumo_files

  for fname in file_names:
    # source and destination files to copy the images
    src= os.path.join(source_dir, fname)
    dest= os.path.join(dest_dir, fname)

    # copy the images from source to dest
    shutil.move(src= src, dst= dest)



class PneumoDataset(Dataset):
  def __init__(self, image_dir, mask_dir, transforms= None, shuffle= False):
    """
    parameter: image_dir - path of the images present in the folder
               mask_dir  - path of corresponding masks present in the folder
               transforms- can provide different geometric transformations to the images and the corresponding masks
    """
    self.image_dir= image_dir
    self.mask_dir= mask_dir
    self.transforms= transforms

    self.images= os.listdir(self.image_dir)
    
    if shuffle== True:
      random.shuffle(self.images)
    

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self,idx):
    image_path= os.path.join(self.image_dir, self.images[idx])
    mask_path= os.path.join(self.mask_dir, self.images[idx])

    # reading image and mask using PIL
    image= Image.open(image_path).convert("P")
    mask= Image.open(mask_path)

    if self.transforms is not None:
      image, mask = self.transforms((image, mask))
            
    image = FT.to_tensor(image) - 0.5
    
    mask = np.array(mask)
    mask = (torch.tensor(mask) > 128).long() 
    return image, mask

