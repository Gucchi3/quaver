# @file cnn_vit.py
# @brief FP32 の CNN + ViT: 64 個のトークンを作る conv の stem、SimA のブロック 2 つ、線形の分類器（pooling はない）
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from .sima_block import SimaBlock



class CnnVit(nn.Module):
    '''
    #* FP32 の CNN + ViT。BatchNorm 付きの 3x3 の conv が 3 つ（16、32、64 チャネルで、stride 2 で 8 x 8 まで小さくする）、64 個のトークンに SimA のブロックが 2 つ、すべてのトークンを読む線形の分類器でできている。
    #? main.py が最初にこれを学習し、BatchNorm を畳み込んで、各 QuantCnnVit に読み込む（層の名前は同じ）。pooling はなく、分類器がすべてのトークンを読む。
    #! uint8 の画像を受け取り、255 で割るだけにする（QuantCnnVit の入力と同じ）。最後の conv（embed）には ReLU がないので、トークンは負にもなる。
    '''
    
    def __init__(self) -> None:
        '''
        #* stem、ブロック、分類器を作る。
        '''
        
        # stem: 32 x 32 で 16 チャネル、16 x 16 で 32 チャネル、そしてトークン（8 x 8 で 64 チャネル）
        super().__init__()
        self.conv1    = nn.Conv2d(3 , 16, 3, 1, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(16)
        self.conv2    = nn.Conv2d(16, 32, 3, 2, 1, bias=False)
        self.bn2      = nn.BatchNorm2d(32)
        self.embed    = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.embed_bn = nn.BatchNorm2d(64)
        
        # SimA のブロック 2 つ（16 次元の head が 4 つ、128 の MLP）と、64 x 8 x 8 個の値を読む分類器
        self.blocks     = nn.ModuleList([SimaBlock(64, 4, 16, 16, 128) for _ in range(2)])
        self.classifier = nn.Linear(64 * 8 * 8, 10)
        
        return None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        #* uint8 の画像の logits を返す。
        '''
        
        x = torch.relu(self.bn1(self.conv1(images.float() / 255)))
        x = torch.relu(self.bn2(self.conv2(x))                   )
        x = self.embed_bn(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.classifier(x.flatten(1))
