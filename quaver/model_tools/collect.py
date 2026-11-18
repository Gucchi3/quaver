# @file collect.py
# @brief モデルの中の量子化器と scaler を集める
# @author Kohei Yamaguchi, Claude

from torch import nn

from ..quantizers import Quantizer



def collect_quantizers(model: nn.Module) -> list[tuple[str, Quantizer]]:
    '''
    #* モデルの中のすべての Quantizer を、名前と一緒に集める。
    '''
    
    return [(name, module) for name, module in model.named_modules() if isinstance(module, Quantizer)]


def collect_scalers[T: nn.Module](model: nn.Module, kind: type[T]) -> list[tuple[str, T]]:
    '''
    #* モデルの中の、1 つのクラスのすべての scaler を、名前と一緒に集める。
    #? kind はクラスで、RangeScaler や LsqScaler など。nn.Module を渡すと、すべての scaler（learnable の属性を持つすべての module）を集める。
    '''
    
    return [(name, module) for name, module in model.named_modules() if isinstance(module, kind) and hasattr(module, "learnable")]
