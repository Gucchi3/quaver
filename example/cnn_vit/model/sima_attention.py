# @file sima_attention.py
# @brief 各 1x1 conv のあとに BatchNorm を置いた（LayerNorm の代わり）FP32 の SimA の attention
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

NORM_MIN = torch.finfo(torch.float32).eps



class SimaAttention(nn.Module):
    '''
    #* FP32 の SimA の attention。Q、K、V は 1x1 conv で作り、Q と K はトークンごとに L1 正規化し、(Q K^T / sqrt(key_dim)) V を求めて、1x1 の projection を通す（論文と同じく ReLU6 はない）。
    #? QuantSimaAttention の float の計算で、LayerNorm の代わりに各 conv のあとに BatchNorm を置く（LeViT と同じ）。fold_batch_norms() がそれを conv に畳み込む。
    #! 各 conv は自分の BatchNorm のすぐ前に置き、conv の名前を QuantSimaAttention と同じにしているので、畳み込んだ重みをそのまま読み込める。
    '''
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int) -> None:
        '''
        #* 4 つの 1x1 conv と、その BatchNorm を作る。
        #? projection の BatchNorm は 0 から始めるので、枝の出力は 0 から始まり、ブロックは恒等写像から始まる。
        '''
        
        # 設定
        super().__init__()
        self.heads     = heads
        self.key_dim   = key_dim
        self.value_dim = value_dim
        
        # Q、K、V、projection の 1x1 conv と、それぞれの BatchNorm
        self.query      = nn.Conv2d(channels         , heads * key_dim  , 1, bias=False)
        self.query_bn   = nn.BatchNorm2d(heads * key_dim)
        self.key        = nn.Conv2d(channels         , heads * key_dim  , 1, bias=False)
        self.key_bn     = nn.BatchNorm2d(heads * key_dim)
        self.value      = nn.Conv2d(channels         , heads * value_dim, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(heads * value_dim)
        self.project    = nn.Conv2d(heads * value_dim, channels         , 1, bias=False)
        self.project_bn = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.project_bn.weight)
        
        return None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        #* (B, C, H, W) に attention をかける。H x W 個の位置がトークンになる。
        '''
        
        # Q、K、V を head に分けて (B, heads, tokens, dims) にし、Q と K をトークンごとに L1 正規化する
        batch, _, height, width = x.shape
        parts                   = ((self.query, self.query_bn), (self.key, self.key_bn), (self.value, self.value_bn))
        query, key, value       = [norm(conv(x)).reshape(batch, self.heads, -1, height * width).transpose(-1, -2) for conv, norm in parts]
        query                   = query / query.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
        key                     = key / key.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
        
        # (Q K^T / sqrt(key_dim)) V と projection
        output = (query @ key.transpose(-1, -2)) * self.key_dim**-0.5 @ value
        output = output.transpose(-1, -2).reshape(batch, self.heads * self.value_dim, height, width)
        return self.project_bn(self.project(output))
