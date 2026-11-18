# @file round.py
# @brief 正確な丸めの関数
# @author Kohei Yamaguchi, Claude

##########################################################
'''
丸め方を足すとき
1. 丸めたテンソルを返す round_xxx(x) を書く。
2. ROUND_REGISTRY に "名前: round_xxx" を足す。
3. types.py の RoundMode に名前を足す。
'''
##########################################################

from collections.abc import Callable

import torch



def round_half_away(x: torch.Tensor) -> torch.Tensor:
    '''
    #* 一番近い整数に丸める。ちょうど半分は 0 から遠い方へ丸める。
    #! どの float でも正確である。floor(|x| + 0.5) は正確でないので使わない。
    '''
    
    whole = torch.trunc(x)
    return whole + torch.trunc(2.0 * (x - whole))


def round_half_even(x: torch.Tensor) -> torch.Tensor:
    '''
    #* 一番近い整数に丸める。ちょうど半分は偶数の方へ丸める。
    '''
    
    return torch.round(x)


def round_half_up(x: torch.Tensor) -> torch.Tensor:
    '''
    #* 一番近い整数に丸める。ちょうど半分は上（+inf の方）へ丸める。
    #! どの float でも正確である。floor(x + 0.5) は正確でないので使わない。
    '''
    
    floor = torch.floor(x)
    return floor + (x - floor >= 0.5).to(x.dtype)


def round_floor(x: torch.Tensor) -> torch.Tensor:
    '''
    #* 算術右シフトと同じく、下へ丸める。
    '''
    
    return torch.floor(x)


ROUND_REGISTRY: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "half_away": round_half_away,
    "half_even": round_half_even,
    "half_up"  : round_half_up,
    "floor"    : round_floor,
}


def build_round(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    '''
    #* 名前の丸めの関数を返す。
    #! 新しい丸め方は ROUND_REGISTRY に足す。
    '''
    
    if name not in ROUND_REGISTRY:
        raise KeyError(f"Unknown rounding mode {name!r}. Available: {', '.join(ROUND_REGISTRY)}")
    return ROUND_REGISTRY[name]
