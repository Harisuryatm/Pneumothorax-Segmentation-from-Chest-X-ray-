# importing library
import torch
import torch.nn as nn
from torchvision.transforms import functional as F

# Repeatitive convolution layers , so creating class that just do the job of calling it and access doube convolution layers
class ConvLayers(nn.Module):
  def __init__(self,in_channels, out_channels):
    super(ConvLayers,self).__init__()
    self.conv= nn.Sequential(
        nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= 3, stride= 1, padding= 1, bias= False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace= True),
        nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, stride= 1, padding= 1, bias= False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace= True)
    )
  
  def forward(self,x):
    return self.conv(x)



class UNET(nn.Module):
  def __init__(self, in_channels= 1, out_channels= 1, features= [64, 128, 256, 512]):
    super(UNET,self).__init__()
    self.upward= nn.ModuleList()
    self.downward= nn.ModuleList()
    self.pool= nn.MaxPool2d(kernel_size= 2, stride= 2)

    # downward convolutions of Unet: Conv2d
    for feature in features:
      self.downward.append(ConvLayers(in_channels= in_channels, out_channels= feature))
      in_channels= feature
    
    # Upward convolutions of Unet: ConvTranspose2d
    for feature in reversed(features):
      self.upward.append(
          nn.ConvTranspose2d(in_channels= feature*2, out_channels= feature, kernel_size= 2, stride= 2)
      )
      self.upward.append(ConvLayers(in_channels= feature*2, out_channels= feature))
    
    # lowest layer which is present below containing 1024 channels
    self.baseLayer= ConvLayers(in_channels= features[-1], out_channels= features[-1]*2)

    # final output conv layer
    self.outputLayer= nn.Conv2d(in_channels= features[0], out_channels= out_channels, kernel_size=1)

  def forward(self, x):
    # storing all the skip connection in the list
    skip_connections= []

    # forwarding through downward layers
    for down in self.downward:
      x= down(x)
      
      # adding skip connections
      skip_connections.append(x)
      x= self.pool(x)
    
    # forwarding through the base layer
    x= self.baseLayer(x)
    
    # reversing the skip connections list to concat it easily
    skip_connections= skip_connections[::-1]
    
    # forwarding through upward layers
    for idx in range(0, len(self.upward), 2):
      x= self.upward[idx](x)
      
      # concatenating the skip connections
      skip_connection= skip_connections[idx//2]

      # Resolving problem of MaxPooling during odd number of height and width 
      if x.shape != skip_connection.shape: 
        x= F.resize(x, size= skip_connection.shape[2:])
      
      
      concat_skip= torch.cat((skip_connection,x), dim= 1)
      
      x= self.upward[idx+1](concat_skip)
    # final output layer
    return self.outputLayer(x)
    
  
