# @file calibrate.py
# @brief 学習の前に、または PTQ のために、活性化の範囲を測る
# @author Kohei Yamaguchi, Claude

from collections.abc import Callable, Iterable
from torch           import nn

import torch

from .collect import collect_scalers
from .switch  import switch_quantizers



def calibrate_model(model: nn.Module, batches: Iterable, forward: Callable | None = None) -> None:
    '''
    #* モデルに float で（量子化を止めて）batch を通し、scaler に float の値の範囲を覚えさせる（PyTorch の observer と同じ）。
    #? PTQ に使うか、QAT の前によい scale から始めるために使う。LSQ の step は最初の batch から始まる。float_mode のフラグを持つ層は float で計算するので、どの値も丸めも切り捨てもしない。
    #! batch が入力のテンソル 1 つでないときは、forward(model, batch) を渡す。
    '''
    
    # eval mode で測るが、scaler は更新させる（switch_quantizers() は層を float のモードにもする）
    was_training = model.training
    model.eval()
    for _, scaler in collect_scalers(model, nn.Module):
        scaler.train()
    switch_quantizers(model, False)
    
    # 勾配なしで batch を通す
    with torch.no_grad():
        for batch in batches:
            if forward is None:
                model(batch)
            else:
                forward(model, batch)
    
    # すべてを元に戻す
    switch_quantizers(model, True)
    model.train(was_training)
    
    return None
