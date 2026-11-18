# @file sima_block.py
# @brief SimA の attention と 1x1 conv の MLP を持ち、LayerNorm の代わりに BatchNorm を使う FP32 の transformer のブロック
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from .sima_attention import SimaAttention



class SimaBlock(nn.Module):
    '''
    #* (B, C, H, W) を受け取る FP32 の transformer のブロック。x + attention(x)、次に x + MLP(x) を求める。MLP は、間に ReLU を挟んだ 2 つの 1x1 conv。
    #? 各 conv は LayerNorm の代わりに自分の BatchNorm を持ち（LeViT と同じ）、fold_batch_norms() がそれを conv に畳み込む。
    #! 層の名前は QuantSimaBlock と同じなので、畳み込んだ重みをそのまま読み込める。
    '''
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int, hidden: int) -> None:
        '''
        #* attention と、MLP の 2 つの conv を、それぞれの BatchNorm と一緒に作る。
        #? MLP の最後の BatchNorm は 0 から始めるので、ブロックは恒等写像から始まる。
        '''
        
        super().__init__()
        self.attention = SimaAttention(channels, heads, key_dim, value_dim)
        self.expand    = nn.Conv2d(channels, hidden  , 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden)
        self.shrink    = nn.Conv2d(hidden  , channels, 1, bias=False)
        self.shrink_bn = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.shrink_bn.weight)
        
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        #* ブロックを動かす: attention と MLP の出力を、それぞれ残差に足す。
        '''
        
        x = x + self.attention(x)
        return x + self.shrink_bn(self.shrink(torch.relu(self.expand_bn(self.expand(x)))))
