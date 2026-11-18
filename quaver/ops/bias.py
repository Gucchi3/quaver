# @file bias.py
# @brief 累積の scale のバイアス（INT32、またはブロックごとの scale のテンソルの wide の累積）
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import RoundMode, format_quantize, format_to_integer, ste_attach
from ..scales    import QuantParams
from .accumulate import accumulate_format, accumulate_scale



def bias_quantize(bias: torch.Tensor, x_params: QuantParams, w_params: QuantParams, mode: RoundMode) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* バイアスを、累積の scale の累積の形式（INT32。ブロックごとの scale のテンソルでは wide）に量子化する。(fake のバイアス, int64 の code) を返す。
    #? training mode は fake のバイアスを足す。integer mode は code を累積に足す。
    #! バイアスの勾配はそのまま通し、scale には勾配を与えない。
    '''
    
    scale         = accumulate_scale(x_params, w_params).detach().double()
    number_format = accumulate_format(x_params, w_params)
    codes         = format_to_integer(format_quantize(bias.detach().double() / scale, number_format, mode), number_format)
    return ste_attach(bias, (codes.double() * scale).float()), codes
