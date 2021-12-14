# importing library
import os
import shutil
import pandas as pd

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
