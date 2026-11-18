# @file switch.py
# @brief 量子化と integer mode を入れたり切ったりする
# @author Kohei Yamaguchi, Claude

from torch import nn

from .collect import collect_quantizers



def switch_quantizers(model: nn.Module, enabled: bool) -> int:
    '''
    #* すべての Quantizer の量子化を入れるか切る。切っている間は、float_mode のフラグを持つすべての層を float のモードにする。変わった量子化器の数を返す。
    #? 切っているときは、層は float で計算し、量子化器は scaler の更新を続ける（PyTorch の observer と同じ。calibrate_model() がこれを使う）。
    #! 切り替えたら、CUDA graph を記録し直す。層に共通のクラスはないので、フラグは setattr() で決める。
    '''
    
    # すべての量子化器を入れるか切る
    quantizers = collect_quantizers(model)
    for _, quantizer in quantizers:
        quantizer.enabled = enabled
    
    # 量子化を切っている間、層は float で計算する
    for module in model.modules():
        if hasattr(module, "float_mode"):
            setattr(module, "float_mode", not enabled)
    return len(quantizers)


def switch_integer_mode(model: nn.Module, enabled: bool) -> int:
    '''
    #* integer_mode のフラグを持つすべての層を integer mode にするか、training mode に戻す。変わった数を返す。
    #? integer mode では、層は整数の code を受け取って返し、デバイスとまったく同じに計算する。
    #! 層に共通のクラスはないので、フラグは setattr() で決める。
    '''
    
    layers = [module for module in model.modules() if hasattr(module, "integer_mode")]
    for layer in layers:
        setattr(layer, "integer_mode", enabled)
    return len(layers)
