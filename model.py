import torch
import torch.nn as nn
from torchvision import models
from AdaIN import AdaIN

class AdaINStyle(nn.Module):
    def __init__(self):
        super().__init__()
        # For encoder load pretrained VGG19 model and remove layers upto relu4_1
        self.vgg = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
        self.vgg = nn.Sequential(*list(self.vgg.features.children())[:21])
        # Create AdaIN layer
        self.ada = AdaIN()
        # Use Sequential to define decoder [Just reverse of vgg with pooling replaced by nearest neigbour upscaling]
        self.dec = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect' ),
        nn.ReLU(),
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU(),
        nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
        nn.ReLU() #Maybe change to a sigmoid to get into 0,1 range?
        )

    def forward(self, c, s):
        """ x is a image containing content information, y is an image
        containing style information"""
        # Compute content and style embeddings
        self.c_emb = self.vgg(c)
        self.s_emb = self.vgg(s)
        # Use AdaIN layer to make the mean and variance of c_emb (content) into that of s_emb (style)
        self.t = self.ada(self.c_emb,self.s_emb)
        return self.dec(self.t)
