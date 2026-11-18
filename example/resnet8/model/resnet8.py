# @file resnet8.py
# @brief CIFAR-10 用の FP32 の ResNet-8（MLPerf Tiny）
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F

import torch

from .basic_block import BasicBlock



class Resnet8(nn.Module):
    '''
    #* FP32 の ResNet-8。3x3 の stem、16、32、64 チャネルのブロック 3 つ、8x8 の平均の pool、線形の分類器でできている。
    #? main.py が最初にこれを学習し、BatchNorm を畳み込んで、各 QuantResnet8 に読み込む（層の名前は同じ）。
    #! uint8 の画像を受け取り、255 で割るだけにする（QuantResnet8 の入力と同じ）。
    '''
    
    def __init__(self) -> None:
        '''
        #* stem、ブロック、分類器を作る。
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
        #* uint8 の画像の logits を返す。
        '''
        
        x = torch.relu(self.stem_bn(self.stem_conv(images.float() / 255)))
        x = self.stack3(self.stack2(self.stack1(x)))
        return self.classifier(F.avg_pool2d(x, 8).flatten(1))
