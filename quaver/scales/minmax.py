# @file minmax.py
# @brief 各グループの値の範囲を測る
# @author Kohei Yamaguchi, Claude

from fractions import Fraction

import math
import torch

from .group import group_split
from .types import Granularity



def minmax_measure(x: torch.Tensor, granularity: Granularity, percentile: float | None = None, symmetric: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* 各グループの範囲を測り、平らなテンソル (low, high) で返す。
    #? 最大と percentile、対称と非対称、すべての粒度を、この 1 つの関数で扱う。
    #! 対称では |x| を測り、(-high, high) を返す。勾配はない。
    '''
    
    # 対称の範囲では |x| を測る
    groups = group_split(x.detach(), granularity)
    if symmetric:
        groups = groups.abs()
    
    # 範囲全体を取る
    if percentile is None:
        low, high = torch.aminmax(groups, dim=1)
        return (-high, high) if symmetric else (low, high)
    
    # または、一番近い順位で percentile を取る: 上の側は percentile、下の側は 100 - percentile（順位は正確な分数で求める）
    count = groups.shape[1]
    rank  = max(1, math.ceil(Fraction(str(percentile)) / 100 * count))
    high  = groups.kthvalue(rank, dim=1).values
    if symmetric:
        return -high, high
    return groups.kthvalue(count - rank + 1, dim=1).values, high
