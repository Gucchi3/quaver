# @file shared_scaler.py
# @brief 1 回の forward で、複数のテンソルが共有する 1 つの scale
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats import NumberFormat
from .compute  import compute_scale
from .ema      import ema_update
from .group    import group_merge
from .minmax   import minmax_measure
from .types    import Granularity



class SharedScaler(nn.Module):
    '''
    #* テンソル全体で 1 つの scale を、1 回の forward の中で複数のテンソル（残差のストリーム）が共有する。
    #? モデルの forward の初めに begin_forward() を、終わりに end_forward() を呼ぶ。
    #? Kaneko と Meng の scale の共有: 1 つのブロックの残差のストリーム（ブロックの入力、枝の出力、足し算）で SharedScaler を 1 つ使う。ブロックが、自分の forward の前後で両方を呼ぶ。
    #! 間では scale は変わらず、移動平均は end_forward() で 1 回だけ更新する。freeze_step は、RangeScaler と同じく、更新を自動で止める。
    '''
    
    learnable: bool = False
    kept     : torch.Tensor
    seen     : torch.Tensor
    updates  : torch.Tensor
    
    def __init__(self, percentile: float | None = None, momentum: float = 0.95, symmetric: bool = True, freeze_step: int | None = None) -> None:
        '''
        #* 設定を持ち、持っておく範囲を NaN（まだ測っていない）で始める。
        #? freeze_step は、範囲が止まるまでの更新（forward）の回数（None なら止めない）。一度も測っていない範囲は必ず最初の更新を受け入れる。
        '''
        
        super().__init__()
        self.percentile  = percentile
        self.momentum    = momentum
        self.symmetric   = symmetric
        self.frozen      = False
        self.freeze_step = freeze_step
        self.register_buffer("kept"   , torch.full((2, 1), torch.nan)     )
        self.register_buffer("seen"   , torch.full((2, 1), torch.nan)     , persistent=False)
        self.register_buffer("updates", torch.zeros((), dtype=torch.int64))
        
        return None
    
    def begin_forward(self, x: torch.Tensor | None = None) -> None:
        '''
        #* forward を始める: それまでに見た範囲を忘れ、持っておく範囲がまだ決まっていなければ x から決める。
        #? ストリームの最初のテンソル（stem の出力）を渡すので、一番最初のパスも 1 つの scale を使う。
        '''
        
        self.seen.fill_(torch.nan)
        if x is not None:
            current = torch.stack(minmax_measure(x, Granularity(), self.percentile, self.symmetric))
            self.kept.copy_(torch.where(torch.isnan(self.kept), current, self.kept))
        
        return None
    
    def forward(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* 共有の scale と zero point を返し、end_forward() のために x の範囲を覚える。
        #? torch.export が動いている間は、持っておいた範囲だけを使う（書き出すモデルは calibration 済み）ので、scale は x によらず、定数に畳み込まれる。
        '''
        
        # 書き出し中: 持っておいた範囲
        if torch.compiler.is_exporting():
            scale, zero_point = compute_scale(self.kept[0], self.kept[1], number_format, self.symmetric)
            return group_merge(scale, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
        
        # この forward で見た一番広い範囲を覚える
        current = torch.stack(minmax_measure(x, Granularity(), self.percentile, self.symmetric))
        if update and self.training and not self.frozen:
            self.seen.copy_(torch.stack([torch.fmin(self.seen[0], current[0]), torch.fmax(self.seen[1], current[1])]))
        
        # 持っておいた範囲を使う。まだ何もないときは x の範囲を使う
        kept              = torch.where(torch.isnan(self.kept), current, self.kept)
        scale, zero_point = compute_scale(kept[0], kept[1], number_format, self.symmetric)
        return group_merge(scale, x.shape, granularity), group_merge(zero_point, x.shape, granularity)
    
    def end_forward(self) -> None:
        '''
        #* forward を終える: 持っておく範囲を見た一番広い範囲に近づけ、freeze_step 回の更新のあとは古い範囲を使い続ける。
        '''
        
        if self.training and not self.frozen:
            seen   = torch.where(torch.isnan(self.seen), self.kept, self.seen)
            moving = ema_update(self.kept, seen, self.momentum)
            if self.freeze_step is not None:
                moving = torch.where((self.updates < self.freeze_step) | torch.isnan(self.kept), moving, self.kept)
            self.kept.copy_(moving)
            self.updates.add_(1)
        
        return None
