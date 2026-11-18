# @file types.py
# @brief 量子化器の型
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass
from typing      import Protocol

import torch

from ..formats import NumberFormat, RoundMode
from ..scales  import Granularity, GranularityKind, LsqInit



@dataclass(frozen=True)
class QuantizerSpec:
    '''
    #* 設定ファイルから Quantizer を作るための設定を持つ。
    #? build_quantizer() がこれを Quantizer にする。自分の config の値をここに写す。round_mode=None のときは、形式のいつもの丸め（FP は half_even、INT は half_away）を使う。
    #! 活性化には batch_axis（たいてい 0）を決め、LSQ の勾配の scale が 1 つのサンプルを数えるようにする。freeze_step は、その回数の更新のあとで、max、percentile、共有の範囲を止める。
    '''
    
    format     : str              = "int8"
    granularity: GranularityKind  = "tensor"
    axis       : int              = 0
    block_size : int              = 16
    scaler     : str              = "max"
    symmetric  : bool             = True
    round_mode : RoundMode | None = None
    momentum   : float            = 0.95
    percentile : float | None     = 99.9
    pow2       : bool             = False
    lsq_init   : LsqInit          = "mse"
    batch_axis : int | None       = None
    freeze_step: int | None       = None


class Scaler(Protocol):
    '''
    #* scaler が持つものを表す: learnable と、scale と zero point を返す呼び出し。
    #? この形で自分の scaler を書けば、Quantizer に渡せる。
    '''
    
    learnable: bool
    
    def __call__(self, x: torch.Tensor, number_format: NumberFormat, granularity: Granularity, update: bool) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        #* scale と zero point を返す。どちらも x に broadcast できる。
        #! update=False のときは、どの状態も変えてはならない。
        '''
        
        ...
