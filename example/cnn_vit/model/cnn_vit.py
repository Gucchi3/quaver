# @file cnn_vit.py
# @brief FP32 CNN + ViT: a conv stem that makes 64 tokens, two SimA blocks, and a linear classifier (no pooling)
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from .sima_block import SimaBlock



class CnnVit(nn.Module):
    '''
    #* FP32 CNN + ViT: three 3x3 convs with BatchNorm (16, 32, and 64 channels, down to 8 x 8 with stride 2), two SimA blocks on the 64 tokens, and a linear classifier on all the tokens.
    #? main.py trains it first, and then folds the BatchNorms and loads it into each QuantCnnVit (the layer names are the same). There is no pooling: the classifier reads every token.
    #! It takes uint8 images and only divides them by 255, like the input of QuantCnnVit. The last conv (embed) has no ReLU, so the tokens can be negative.
    '''
    
    def __init__(self) -> None:
        '''
        #* Make the stem, the blocks, and the classifier.
        '''
        
        # Stem: 32 x 32 with 16 channels, 16 x 16 with 32, and the tokens: 8 x 8 with 64
        super().__init__()
        self.conv1    = nn.Conv2d(3 , 16, 3, 1, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(16)
        self.conv2    = nn.Conv2d(16, 32, 3, 2, 1, bias=False)
        self.bn2      = nn.BatchNorm2d(32)
        self.embed    = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.embed_bn = nn.BatchNorm2d(64)
        
        # Two SimA blocks (4 heads of 16, an MLP of 128), and the classifier on the 64 x 8 x 8 values
        self.blocks     = nn.ModuleList([SimaBlock(64, 4, 16, 16, 128) for _ in range(2)])
        self.classifier = nn.Linear(64 * 8 * 8, 10)
        
        return None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        #* Return the logits of uint8 images.
        '''
        
        x = torch.relu(self.bn1(self.conv1(images.float() / 255)))
        x = torch.relu(self.bn2(self.conv2(x))                   )
        x = self.embed_bn(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.classifier(x.flatten(1))
