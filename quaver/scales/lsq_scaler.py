# @file lsq_scaler.py
# @brief 学習する scale（LSQ）と学習する zero point（LSQ+）
# @author Kohei Yamaguchi, Claude

from torch  import nn
from typing import Any

import math
import torch

from ..formats import NumberFormat, ste_round
from .compute  import compute_mse_range, compute_scale
from .group    import group_merge, group_split
from .minmax   import minmax_measure
from .types    import Granularity, LsqInit

STEP_MIN = torch.finfo(torch.float32).eps



class LsqScaler(nn.Module):
    '''
    #* 勾配降下で scale（step）を学習する（LSQ）。非対称なら zero point も学習する（LSQ+）。
    #? Quantizer は lsq_quantize() で、fake の op は params_attach() で、LSQ の勾配を与える。最初の学習の batch が初期値を決める。
    #! step を 0 より上に保つため、optimizer の step のたびに lsq_clamp_steps() を呼ぶ。
    '''
    
    learnable: bool = True
    
    def __init__(self, scale_count: int = 1, symmetric: bool = True, init: LsqInit = "mse") -> None:
        '''
        #* step（最初の batch までは NaN）と、非対称なら zero point を作る。
        #? init: "mse"（二乗誤差が最小。活性化の LSQ+）、"gauss"（平均 +- 3 std。重みの LSQ+）、"lsq"（2 x mean|x| / sqrt(Qp)。LSQ）、"range"（最小と最大）。
        #! "lsq" は対称の step だけに使う。非対称で "lsq" のときは、測った範囲から始める。
        '''
        
        super().__init__()
        self.symmetric   = symmetric
        self.init        = init
        self.initialized = False
        self.step        = nn.Parameter(torch.full((scale_count,), torch.nan))
        self.offset      = None if symmetric else nn.Parameter(torch.zeros(scale_count))
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* 学習した step と zero point（勾配付き）を返す。どちらも x に broadcast できる。
        #! zero point は、straight-through の勾配で整数に丸める。
        '''
        
        # 最初の学習の batch で初期値を決める
        if update and self.training and not self.initialized:
            with torch.no_grad():
                # 範囲: 測ったもの、二乗誤差が最小のもの（活性化の LSQ+）、または平均 +- 3 std（重みの LSQ+）
                groups    = group_split(x.detach(), granularity)
                low, high = minmax_measure(x, granularity, None, self.symmetric)
                if self.init == "mse":
                    low, high = compute_mse_range(groups, low, high, number_format, self.symmetric)
                elif self.init == "gauss":
                    center, spread = groups.mean(dim=1), 3 * groups.std(dim=1, correction=0)
                    low, high      = center - spread, center + spread
                
                # 範囲を step と zero point に変える。論文の対称の step: LSQ は 2 x mean|x| / sqrt(Qp) を使い、LSQ+ は大きい方の端を端の code（符号付きの INT なら 2^(b-1)）に合わせる
                scale, zero_point = compute_scale(low, high, number_format, self.symmetric)
                if self.symmetric and self.init == "lsq":
                    size  = groups.abs() if number_format.signed else groups.clamp_min(0.0)
                    scale = (2.0 * size.mean(dim=1) / math.sqrt(number_format.max_value)).clamp_min(STEP_MIN)
                elif self.symmetric and self.init == "gauss":
                    scale = (torch.maximum(low.abs(), high.abs()) / max(-number_format.min_value, number_format.max_value)).clamp_min(STEP_MIN)
                
                # それを持つ
                self.step.copy_(scale)
                if self.offset is not None:
                    self.offset.copy_(zero_point)
            self.initialized = True
        
        # step を 0 より上に、zero point を整数の code に保つ
        step = self.step.clamp_min(STEP_MIN)
        if self.offset is None:
            zero_point = torch.zeros_like(step)
        else:
            zero_point = ste_round(self.offset, "half_even").clamp(number_format.min_value, number_format.max_value)
        return group_merge(step, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
    
    def _load_from_state_dict(self, state_dict: dict[str, Any], prefix: str, local_metadata: dict[str, Any], strict: bool, missing_keys: list[str], unexpected_keys: list[str], error_msgs: list[str]) -> None:
        '''
        #* 保存した step を読み込んだあとで、scaler を始まった状態にする。
        #? PyTorch が load_state_dict() から呼ぶ。
        #! ここでは GPU から値を読む。読み込みは学習の step の一部ではないので、それでよい。
        '''
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.initialized = bool(torch.isfinite(self.step).all())
        
        return None
