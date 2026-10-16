# @file sima_attention.py
# @brief FP32 SimA attention with a BatchNorm after each 1x1 conv (in place of LayerNorm)
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

NORM_MIN = torch.finfo(torch.float32).eps



class SimaAttention(nn.Module):
    '''
    #* FP32 SimA attention: Q, K, and V from 1x1 convs, Q and K L1-normalized for each token, (Q K^T / sqrt(key_dim)) V, and a 1x1 projection (no ReLU6, like the paper).
    #? It is the float math of QuantSimaAttention, with a BatchNorm after each conv in place of LayerNorm (like LeViT). fold_batch_norms() folds them into the convs.
    #! Each conv comes just before its BatchNorm, and the convs have the names of QuantSimaAttention, so the folded weights load into it.
    '''
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int) -> None:
        '''
        #* Make the four 1x1 convs and their BatchNorms.
        #? The BatchNorm of the projection starts at 0, so the branch starts at 0 and the block starts as the identity.
        '''
        
        # Settings
        super().__init__()
        self.heads     = heads
        self.key_dim   = key_dim
        self.value_dim = value_dim
        
        # The 1x1 convs of Q, K, V, and the projection, each with its BatchNorm
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
        #* Run the attention on (B, C, H, W), with the H x W positions as the tokens.
        '''
        
        # Q, K, and V cut into heads as (B, heads, tokens, dims), and Q and K L1-normalized for each token
        batch, _, height, width = x.shape
        parts                   = ((self.query, self.query_bn), (self.key, self.key_bn), (self.value, self.value_bn))
        query, key, value       = [norm(conv(x)).reshape(batch, self.heads, -1, height * width).transpose(-1, -2) for conv, norm in parts]
        query                   = query / query.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
        key                     = key / key.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
        
        # (Q K^T / sqrt(key_dim)) V and the projection
        output = (query @ key.transpose(-1, -2)) * self.key_dim**-0.5 @ value
        output = output.transpose(-1, -2).reshape(batch, self.heads * self.value_dim, height, width)
        return self.project_bn(self.project(output))
