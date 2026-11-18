# @file range_scaler.py
# @brief 範囲（最大か percentile）の移動平均から求める scale
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats import NumberFormat
from .compute  import compute_scale
from .ema      import ema_update
from .group    import group_merge
from .minmax   import minmax_measure
from .types    import Granularity



class RangeScaler(nn.Module):
    '''
    #* 学習中は範囲を移動平均で追い、推論ではそれを使い続ける。
    #? percentile=None のときは最大を追う（"max"）。99.9 なら 99.9 番目の percentile を追う（"percentile"）。
    #! frozen = True ですぐに更新を止める（変えたら CUDA graph を記録し直す）。freeze_step は、その回数の更新のあとで自動で止め、記録し直さなくてよい。
    '''
    
    learnable: bool = False
    kept     : torch.Tensor
    updates  : torch.Tensor
    
    def __init__(self, percentile: float | None = None, momentum: float = 0.95, scale_count: int = 1, symmetric: bool = True, freeze_step: int | None = None) -> None:
        '''
        #* 設定を持ち、持っておく範囲を NaN（まだ測っていない）で始める。
        #? scale_count は scale の数: テンソルなら 1、チャネルごとならチャネルの数。freeze_step は、範囲が止まるまでの更新の回数（学習の step ごとに 1 回）。
        #! 回数はデバイスの上にあるので、止めるときに GPU の同期はない。一度も測っていない範囲は必ず最初の更新を受け入れるので、freeze_step=0 なら最初の batch の範囲を使い続ける。
        '''
        
        super().__init__()
        self.percentile  = percentile
        self.momentum    = momentum
        self.symmetric   = symmetric
        self.frozen      = False
        self.freeze_step = freeze_step
        self.register_buffer("kept"   , torch.full((2, scale_count), torch.nan))
        self.register_buffer("updates", torch.zeros((), dtype=torch.int64)     )
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* 学習中は範囲を更新してから、持っておいた範囲から scale と zero point を返す。
        #! 最初の学習の batch の前は範囲が NaN なので、先に学習するか calibrate_model() を動かす。
        '''
        
        # 学習中は範囲の移動平均を更新し、freeze_step 回の更新のあとは古い範囲を使い続ける
        if update and self.training and not self.frozen:
            current = torch.stack(minmax_measure(x, granularity, self.percentile, self.symmetric))
            moving  = ema_update(self.kept, current, self.momentum)
            if self.freeze_step is not None:
                moving = torch.where((self.updates < self.freeze_step) | torch.isnan(self.kept), moving, self.kept)
            self.kept.copy_(moving)
            self.updates.add_(1)
        
        # 持っておいた範囲を scale に変える
        scale, zero_point = compute_scale(self.kept[0], self.kept[1], number_format, self.symmetric)
        return group_merge(scale, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
