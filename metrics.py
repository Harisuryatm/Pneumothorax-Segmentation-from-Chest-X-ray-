import torch
from torch.nn import functional as F

def IoU(y_true, y_pred):
    """ Jaccard a.k.a IoU score for batch of images
    """
    
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    intersection = (y_true_flat * y_pred_flat).sum(1)
    union = ((y_true_flat + y_pred_flat) > 0.0).float().sum(1)
    
    score = (intersection) / (union + eps)
    score = score.sum() / num
    return score
  
def dice(y_true, y_pred):
    """ Dice a.k.a f1 score for batch of images
    """
    num = y_true.size(0)
    eps = 1e-7
    
    y_true_flat = y_true.view(num, -1)
    y_pred_flat = y_pred.view(num, -1)
    intersection = (y_true_flat * y_pred_flat).sum(1)
    
    score =  (2 * intersection) / (y_true_flat.sum(1) + y_pred_flat.sum(1) + eps)
    score = score.sum() / num
    return score



def check_accuracy(loader,valid_imgs_length, model,loss_fn, device= "cuda"):
  """
  Returns the accuracy, IoU score and dice score of the predictions to evaluate the model performance
  """
  val_loss = 0.0
  val_iou = 0.0
  val_dice = 0.0

  model.eval()

  for images, masks in loader:
    num= images.size(0)

    images= images.to(device= device)
    masks= masks.to(device= device)

    with torch.no_grad():
      valid_outputs= model(images)
      preds= F.log_softmax(valid_outputs, dim=1)
      val_loss += loss_fn(preds, masks).item() * num

      valid_outputs= torch.argmax(preds, dim= 1)
      valid_outputs= valid_outputs.float()
      masks= masks.float()

      val_iou += IoU(masks, valid_outputs.float()).item() * num
      val_dice += dice(masks, valid_outputs).item() * num
    
    print(".", end="")
  
  val_loss = val_loss / valid_imgs_length
  val_iou = val_iou / valid_imgs_length
  val_dice = val_dice / valid_imgs_length

  return val_loss,val_iou, val_dice
