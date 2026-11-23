# @file quant_add.py
# @brief 量子化した残差の足し算のサンプル
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats    import RoundMode, build_format
from ..ops        import mode_ops
from ..quantizers import Quantizer, Scaler
from ..scales     import Granularity, QuantParams, RangeScaler



class QuantAdd(nn.Module):
    '''
    #* 残差の足し算: デバイスと同じく、両方の入力を出力の scale に直して足し、和を 1 回だけ丸めて、切り詰める。
    #? forward(a, a_params, b, b_params) は (出力, 出力のパラメータ) を返す。出力の scale は、和の範囲に従うか、共有する（output_scaler）。
    #? scale の共有（Kaneko と Meng）: a、b、出力が 1 つの SharedScaler を使うときは、比が 1 になり、足し算は飽和付きのただの整数の足し算になる（M0 = 2^15、n = 15 は正確）。
    '''
    
    def __init__(self, output_format: str = "int8", momentum: float = 0.95, style: str = "mul16_shift", round_mode: RoundMode | None = None, output_symmetric: bool = True, output_scaler: Scaler | None = None) -> None:
        '''
        #* 出力の量子化器を作り、設定を持つ。
        #? output_scaler は、出力に共有の scale（SharedScaler）を与える。None のときは、和に従う RangeScaler を作る。
        #! output_symmetric=False のときは、出力に zero point を持たせる（INT の形式だけ）。
        '''
        
        super().__init__()
        self.output_quantizer = Quantizer(build_format(output_format), RangeScaler(momentum=momentum, symmetric=output_symmetric) if output_scaler is None else output_scaler, Granularity(), round_mode, batch_axis=0)
        self.style            = style
        self.integer_mode     = False
        self.float_mode       = False
        
        return None
    
    def forward(self, a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* 今のモード（fake の値、整数の code、float の値）で a と b を足す。
        '''
        
        ops        = mode_ops(self.integer_mode, self.float_mode)
        out_params = self.output_quantizer.params(a) if self.integer_mode else self.output_quantizer.measure(a + b)
        return ops.add(a, a_params, b, b_params, out_params, self.style), out_params
