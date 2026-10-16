# @file basic_block.py
# @brief FP32 block of ResNet-8: two 3x3 convs with BatchNorm and a residual add
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch



class BasicBlock(nn.Module):
    '''
    #* Two 3x3 convs with BatchNorm, a shortcut (as it is, or a 1x1 conv with a bias), an add, and ReLU.
    #? Same layout as the ResNet-8 of MLPerf Tiny (resnet_v1_eembc).
    #! Each conv comes just before its BatchNorm in the same module, so fold_batch_norms() can fold them.
    '''
    
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        '''
        #* Make the two convs, their BatchNorms, and the shortcut.
        '''
        
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, stride)
        
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        #* Run the block.
        '''
        
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + self.shortcut(x))
