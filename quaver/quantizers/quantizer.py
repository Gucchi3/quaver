# @file quantizer.py
# @brief Quantizer: 数値形式 + scaler + 粒度 + 丸め
# @author Kohei Yamaguchi, Claude

from torch import nn

import math
import torch

from ..formats import NumberFormat, RoundMode, format_round_mode, ste_quantize
from ..scales  import BlockScaler, Granularity, QuantParams, compute_pow2, group_count, params_note
from .lsq      import lsq_grad_scale, lsq_quantize, lsq_scale_gradient
from .types    import Scaler



class Quantizer(nn.Module):
    '''
    #* 数値形式、scaler、粒度、丸め方でテンソルを量子化する。
    #? forward() は training mode（fake quantization）、encode() は integer mode（code）。
    #! enabled = False のときは、forward() は scaler を更新するだけで、x をそのまま返す。
    '''
    
    round_mode: RoundMode
    
    def __init__(self, number_format: NumberFormat, scaler: Scaler, granularity: Granularity = Granularity(), round_mode: RoundMode | None = None, pow2: bool = False, batch_axis: int | None = None) -> None:
        '''
        #* 量子化器の部品を持つ。
        #? round_mode=None のときは、形式のいつもの丸め（FP は half_even、INT は half_away）を使う。pow2=True のときは、scale を 2 のべき乗に切り上げる（shift しかできないデバイス用）。
        #! 活性化の量子化器には batch_axis（サンプルの軸、たいてい 0）を渡す: そうすると、LSQ の論文と同じく、LSQ の勾配の scale が 1 つのサンプルの値を数える。重みは None のままにする。
        '''
        
        # この入口で確かめる: FP の形式には zero point がないので scaler は対称でなければならない。ブロックの scale は自分のグリッドの上にあり、2 のべき乗ではない
        if number_format.kind == "fp" and not getattr(scaler, "symmetric", True):
            raise ValueError(f"Asymmetric quantization works only for INT formats, not for {number_format.name}. Use a symmetric scaler.")
        if pow2 and isinstance(scaler, BlockScaler):
            raise ValueError("pow2 does not work with BlockScaler: its block scales are on the grid of their own format.")
        
        # 部品を持つ
        super().__init__()
        self.number_format = number_format
        self.scaler        = scaler
        self.granularity   = granularity
        self.round_mode    = format_round_mode(number_format, round_mode)
        self.pow2          = pow2
        self.batch_axis    = batch_axis
        self.enabled       = True
        
        return None
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* training mode: fake quantize した x とそのパラメータを返す。
        #! float32 で計算して返す。
        '''
        
        # scaler から scale と zero point を受け取る
        x      = x.float()
        params = self._params(x, update=True)
        if not self.enabled:
            return x, params
        
        # 学習する scale には LSQ の勾配を、ほかには straight-through の勾配を与える
        if self.scaler.learnable:
            return lsq_quantize(x, params.scale, params.zero_point, self.number_format, self.round_mode, self._grad_scale(x)), params
        return ste_quantize(x, params.scale, params.zero_point, self.number_format, self.round_mode), params
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* integer mode: x の整数の code（int64）とそのパラメータを返す。
        #? 実数の値 = (code - zero_point) x params.code_scale。code はデバイスの op torch.ops.quaver.quantize から来るので、torch.export はそれを 1 つのノードとして残す。
        #! scaler は変えない。
        '''
        
        with torch.no_grad():
            params = self._params(x.float(), update=False)
            codes  = torch.ops.quaver.quantize(x.float(), params.scale, params.zero_point, self.number_format.name, self.round_mode)
        return params_note(codes, params), params
    
    def params(self, x: torch.Tensor) -> QuantParams:
        '''
        #* scaler を変えずに、x の形のテンソルに対する今のパラメータを返す。
        #? 毎回測る scaler（dynamic、nvfp4）でなければ、x は形を与えるだけ。
        '''
        
        with torch.no_grad():
            return self._params(x.float(), update=False)
    
    def measure(self, x: torch.Tensor) -> QuantParams:
        '''
        #* scaler に x を見せ（学習中は更新する）、量子化せずにパラメータを返す。
        #? この量子化器でなく、整数の op が丸める出力に、層が使う。学習する step には LSQ の勾配の scale が付いてくるので、その op で学習できる。
        '''
        
        params = self._params(x.detach().float(), update=True)
        if not self.scaler.learnable:
            return params
        return QuantParams(lsq_scale_gradient(params.scale, self._grad_scale(x)), params.zero_point, params.number_format)
    
    def _params(self, x: torch.Tensor, update: bool) -> QuantParams:
        '''
        #* scaler に scale と zero point を聞き、pow2 が入っていれば scale を 2 のべき乗に丸める。
        #? forward()、encode()、params()、measure() が使う。BlockScaler は、ブロックの scale の整数の部分も返す。
        '''
        
        # ブロックの scale は、正確な整数の計算のために、整数の部分と一緒に来る
        if isinstance(self.scaler, BlockScaler):
            scale, zero_point, block = self.scaler.split(x, self.number_format, self.granularity, update)
            return QuantParams(scale, zero_point, self.number_format, block)
        
        # ほかの scaler は scale と zero point を返す
        scale, zero_point = self.scaler(x, self.number_format, self.granularity, update)
        if self.pow2:
            scale = scale + (compute_pow2(scale.detach()) - scale).detach()
        return QuantParams(scale, zero_point, self.number_format)
    
    def _grad_scale(self, x: torch.Tensor) -> float:
        '''
        #* x に対する LSQ の勾配の scale を返す: 1 / sqrt(N x Qp)。N は 1 つの scale を共有する値の数。
        #? forward() と measure() が使う。batch_axis があるときは N を 1 つのサンプルで数えるので、勾配は batch の大きさで変わらない。
        '''
        
        shape = list(x.shape)
        if self.batch_axis is not None:
            shape[self.batch_axis] = 1
        return lsq_grad_scale(math.prod(shape) // group_count(torch.Size(shape), self.granularity), self.number_format)
