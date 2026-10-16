# @file sima_block.py
# @brief FP32 transformer block with SimA attention and an MLP of 1x1 convs, with BatchNorm in place of LayerNorm
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from .sima_attention import SimaAttention



class SimaBlock(nn.Module):
    '''
    #* FP32 transformer block on (B, C, H, W): x + attention(x), then x + MLP(x), where the MLP is two 1x1 convs with ReLU between them.
    #? Each conv has its own BatchNorm in place of LayerNorm (like LeViT), and fold_batch_norms() folds them into the convs.
    #! The layer names are those of QuantSimaBlock, so the folded weights load into it.
    '''
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int, hidden: int) -> None:
        '''
        #* Make the attention and the two convs of the MLP with their BatchNorms.
        #? The last BatchNorm of the MLP starts at 0, so the block starts as the identity.
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
        #* Run the block: the attention and the MLP, each added to the residual.
        '''
        
        x = x + self.attention(x)
        return x + self.shrink_bn(self.shrink(torch.relu(self.expand_bn(self.expand(x)))))
