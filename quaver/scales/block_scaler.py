# @file block_scaler.py
# @brief ブロックごとの scale の形式（NVFP4、整数のブロックの scale）の 2 段の scale: ブロックごとに 1 つと、テンソルで 1 つ
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats       import NumberFormat, build_format, format_to_integer, integer_format
from .compute        import SCALE_MIN, compute_block
from .dynamic_scaler import DynamicScaler
from .group          import group_merge, group_split
from .types          import BlockScale, Granularity

# 一番大きな値が 1 の形式: これで scale を頼まれた scaler は、範囲そのものを返す
UNIT_FORMAT = integer_format(1, signed=False)



class BlockScaler(nn.Module):
    '''
    #* ブロックごとの scale の形式の 2 つの scale を作る: ブロックごとに 1 つ（scale_format に丸める）と、テンソルで 1 つ。
    #? scale_format "ue4m3"（符号なしの E4M3、448 まで）で NVFP4 に、"uint6" か "uint10" で整数のブロックの scale（VS-Quant）になる。Granularity("block", axis, 16) と一緒に使う。
    #? tensor_scaler はテンソルの範囲を測る: DynamicScaler（標準）は毎回測るので、重みに使う。RangeScaler はテンソルの scale を静的に保つので、活性化に使う。
    #! ブロックの scale は、いつもブロックごとに決まる。zero point はいつも 0。
    '''
    
    learnable: bool = False
    
    def __init__(self, scale_format: str = "ue4m3", tensor_scaler: nn.Module | None = None) -> None:
        '''
        #* ブロックの scale の形式と、テンソルの scaler を持つ。
        '''
        
        super().__init__()
        self.scale_format  = build_format(scale_format)
        self.tensor_scaler = DynamicScaler() if tensor_scaler is None else tensor_scaler
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* 各値の scale（ブロックの scale x テンソルの scale）と、0 の zero point を返す。
        '''
        
        scale, zero_point, _ = self.split(x, number_format, granularity, update)
        return scale, zero_point
    
    def split(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool) -> tuple[torch.Tensor, torch.Tensor, BlockScale]:
        '''
        #* 各値の scale、zero point 0、ブロックの scale の整数の部分（各値のブロックの scale の code と、テンソルの scale）を返す。
        #? Quantizer がこれを使うので、ブロックごとの scale のテンソルの QuantParams は、ブロックの scale を持つ。
        '''
        
        # テンソルの scale と、scale_format のグリッドの上の各ブロックの scale
        tensor_scale = self.tensor_scale(x, number_format, update)
        block_scale  = compute_block(group_split(x.detach(), granularity).abs().amax(dim=1), tensor_scale, number_format, self.scale_format)
        
        # scale を値に広げ、ブロックの scale の code を持つ
        scale = group_merge(block_scale * tensor_scale                       , x.shape, granularity)
        codes = group_merge(format_to_integer(block_scale, self.scale_format), x.shape, granularity)
        return scale, torch.zeros_like(scale), BlockScale(codes, self.scale_format, tensor_scale)
    
    def tensor_scale(self, x: torch.Tensor, number_format: NumberFormat, update: bool) -> torch.Tensor:
        '''
        #* テンソルの scale を返す: テンソルの scaler の |値| の最大 / (number_format の上限 x 一番大きなブロックの scale)。そうすると、一番大きなブロックが一番上のブロックの scale になる。
        #? 整数の計算で活性化をブロックに量子化する層が、静的なテンソルの scale（テンソルの scaler は RangeScaler）に使う。
        '''
        
        tensor_amax = self.tensor_scaler(x, UNIT_FORMAT, Granularity(), update)[0]
        return (tensor_amax / (self.scale_format.max_value * number_format.max_value)).clamp_min(SCALE_MIN)
