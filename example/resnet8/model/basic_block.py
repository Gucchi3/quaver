# @file basic_block.py
# @brief ResNet-8 の FP32 のブロック: BatchNorm 付きの 3x3 の conv 2 つと、残差の足し算
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch



class BasicBlock(nn.Module):
    '''
    #* BatchNorm 付きの 3x3 の conv 2 つ、shortcut（そのまま、またはバイアス付きの 1x1 conv）、足し算、ReLU でできている。
    #? MLPerf Tiny の ResNet-8（resnet_v1_eembc）と同じ形。
    #! 各 conv は同じ module の中で自分の BatchNorm のすぐ前に置くので、fold_batch_norms() で畳み込める。
    '''
    
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        '''
        #* 2 つの conv、その BatchNorm、shortcut を作る。
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
        #* ブロックを動かす。
        '''
        
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(y + self.shortcut(x))
