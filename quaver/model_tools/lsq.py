# @file lsq.py
# @brief optimizer のための LSQ の道具
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..scales import STEP_MIN, LsqScaler
from .collect import collect_scalers



def lsq_parameters(model: nn.Module) -> list[nn.Parameter]:
    '''
    #* すべての LsqScaler の、学習した step と zero point を集める。
    '''
    
    return [parameter for _, scaler in collect_scalers(model, LsqScaler) for parameter in scaler.parameters()]


def lsq_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    '''
    #* optimizer のグループを作る: ほかのパラメータはすべて weight_decay 付き、LSQ の step と zero point は weight decay なしにする。
    '''
    
    learned = lsq_parameters(model)
    chosen  = {id(parameter) for parameter in learned}
    others  = [parameter for parameter in model.parameters() if id(parameter) not in chosen]
    return [{"params": others, "weight_decay": weight_decay}, {"params": learned, "weight_decay": 0.0}]


def lsq_clamp_steps(model: nn.Module) -> None:
    '''
    #* LSQ の step をすべて、下限より上に戻す。optimizer.step() のあとで呼ぶ。
    #? Adam は step を学習率くらい動かすので、1 回の更新で小さな step が負になることがある。
    '''
    
    with torch.no_grad():
        for _, scaler in collect_scalers(model, LsqScaler):
            scaler.step.clamp_(min=STEP_MIN)
    
    return None
