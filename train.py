# import library
import torch
import torch.optim as optim
from model import UNet
import numpy as np
import time
from torch.nn import functional as F
import torch.optim as optim
from metrics import IoU, dice, check_accuracy


DEVICE= "cuda" if torch.cuda.is_available() == True else "cpu"
scaler= torch.cuda.amp.GradScaler()

train_log_filename = "/content/train-log.txt"
best_val_loss = np.inf


def train_fn(train_loader, valid_loader,train_imgs_length, valid_imgs_length, num_epochs,  model, optimizer, loss_fn):
  """Training of Pneumothorax model"""
  
  hist = []
  
  for epoch in range(num_epochs):
    start_t = time.time()
    print("train phase")
    model.train()
    train_loss = 0.0
    
    for images, masks in train_loader:
      num = images.size(0)
      
      # allocating data to GPU    
      images = images.to(device=DEVICE)
      masks = masks.to(device= DEVICE)
      
      # forward pass
      with torch.cuda.amp.autocast():
        train_outputs= model(images)
        preds = F.log_softmax(train_outputs, dim=1)
        loss = loss_fn(preds, masks)
      
      # backward
      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      
      train_loss += loss.item() * num
      print(".", end="")
    
    train_loss= train_loss / train_imgs_length
    print()
      
      
    
    print("Validation phase")
    val_loss,val_iou, val_dice= check_accuracy(valid_loader,valid_imgs_length, model, loss_fn, device= DEVICE)
    print()

    end_t = time.time()
    spended_t = end_t - start_t

    with open(train_log_filename, "a") as train_log_file:
      report = f"epoch: {epoch+1}/{num_epochs}, time: {spended_t}, train loss: {train_loss}, \n"\
               + f"val loss: {val_loss}, val jaccard: {val_iou}, val dice: {val_dice}"
      
      hist.append({
            "time": spended_t,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_jaccard": val_iou,
            "val_dice": val_dice,
        })

      print(report)
      train_log_file.write(report + "\n")

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint= {"state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict()}
        
        save_checkpoint(checkpoint)
        print("model saved")
        train_log_file.write("model saved\n")
      print()
