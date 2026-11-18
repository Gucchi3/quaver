# @file model_outputs.py
# @brief code とそのパラメータを返す forward() を持つモデルを、torch.export で記録できるようにする
# @author Kohei Yamaguchi, Claude

from torch  import nn
from typing import Any

import torch

from ..scales import QuantParams, params_note



class ModelOutputs(nn.Module):
    '''
    #* モデルを動かし、出力をテンソルの tuple で返す。テンソルはそのまま、組 (codes, QuantParams) は code を返す。
    #? torch.export は出力にテンソルしか受け付けないので、model_export() はこれを記録する。組の code にはパラメータの印（params_note）を付け、書き出しはそれを出力に書く。
    #! モデルの forward() は、テンソル、組、またはテンソルと組の tuple か list を返してよい。
    '''
    
    def __init__(self, model: nn.Module) -> None:
        '''
        #* モデルを持つ。
        '''
        
        super().__init__()
        self.model = model
        
        return None
    
    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        '''
        #* モデルを動かし、テンソルと、印を付けた組の code を、順に返す。
        '''
        
        returned = self.model(*inputs)
        items    = [returned] if isinstance(returned, torch.Tensor) or ModelOutputs.pair(returned) else list(returned)
        return tuple(item if isinstance(item, torch.Tensor) else params_note(*item) for item in items)
    
    @staticmethod
    def pair(item: Any) -> bool:
        '''
        #* item が、code とその QuantParams の組かどうかを返す。
        #? forward() が、モデルが返すものを読むときに使う。
        '''
        
        return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], torch.Tensor) and isinstance(item[1], QuantParams)
