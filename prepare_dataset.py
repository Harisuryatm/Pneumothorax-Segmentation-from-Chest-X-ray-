# importing library
import os
import shutil
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

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
  for both train and validation
  parameters -  dataset   :  csv file containing the image details
                source_dir:  directory from source
                dest_dir  :  directory to move images to destination  
  """
  # files to be stored
  file_names= list(dataset["new_filename"].values)

  for fname in file_names:
    # source and destination files to copy the images
    src= os.path.join(source_dir, fname)
    dest= os.path.join(dest_dir, fname)

    # copy the images from source to dest
    shutil.move(src= src, dst= dest)



class PneumoDataset(Dataset):
  """
  Custom dataset for Pneumothorax Segmentation
  """
  def __init__(self, image_dir, mask_dir, transforms= None):
    """
    parameter: image_dir - path of the images present in the folder
               mask_dir  - path of corresponding masks present in the folder
               transforms- can provide different geometric transformations to the images and the corresponding masks
    """
    self.image_dir= image_dir
    self.mask_dir= mask_dir
    self.transforms= transforms

    self.images= os.listdir(self.image_dir)

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self):
    image_path= os.path.join(self.image_dir, self.images[idx])
    mask_path= os.path.join(self.mask_dir, self.images[idx])

    # reading image and mask using PIL
    image= np.array(Image.open(image_path).convert("RGB"))
    mask= np.array(Image.open(mask_path).convert("L"), dtype= np.float32)
    
    mask[mask== 255.0]= 1.0
    
    if self.transforms is not None:
      albumentations= self.transforms(image= image, mask= mask)
      image= albumentations["image"]
      mask= albumentations["mask"]
    
    return image,mask    
