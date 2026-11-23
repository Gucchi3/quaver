# @file ema.py
# @brief NaN から始まる移動平均
# @author Kohei Yamaguchi, Claude

import torch



def ema_update(running: torch.Tensor, current: torch.Tensor, momentum: float) -> torch.Tensor:
    '''
    #* running と current の移動平均を返す。
    #! running が NaN（一度も測っていない）の所では、current をそのまま返す。torch.where を使うので、GPU の同期はない。
    '''
    
    average = running * momentum + current * (1.0 - momentum)
    return torch.where(torch.isnan(running), current, average)
