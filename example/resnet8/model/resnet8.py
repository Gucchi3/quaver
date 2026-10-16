# @file resnet8.py
# @brief FP32 ResNet-8 for CIFAR-10 (MLPerf Tiny)
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F

import torch

from .basic_block import BasicBlock



class Resnet8(nn.Module):
    '''
    #* FP32 ResNet-8: a 3x3 stem, three blocks with 16, 32, and 64 channels, an 8x8 average pool, and a linear classifier.
    #? main.py trains it first, and then folds the BatchNorms and loads it into each QuantResnet8 (the layer names are the same).
    #! It takes uint8 images and only divides them by 255, like the input of QuantResnet8.
    '''
    
    def __init__(self) -> None:
        '''
        #* Make the stem, the blocks, and the classifier.
        '''
        
        super().__init__()
        self.stem_conv  = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.stem_bn    = nn.BatchNorm2d(16)
        self.stack1     = BasicBlock(16, 16, 1)
        self.stack2     = BasicBlock(16, 32, 2)
        self.stack3     = BasicBlock(32, 64, 2)
        self.classifier = nn.Linear(64, 10)
        
        return None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the logits of uint8 images.
        '''
        
        x = torch.relu(self.stem_bn(self.stem_conv(images.float() / 255)))
        x = self.stack3(self.stack2(self.stack1(x)))
        return self.classifier(F.avg_pool2d(x, 8).flatten(1))
