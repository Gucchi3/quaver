# @file fold.py
# @brief BatchNorm を、その前の conv に畳み込む
# @author Kohei Yamaguchi, Claude

from torch                 import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

import copy

FOLD_PAIRS = ((nn.Conv1d, nn.BatchNorm1d), (nn.Conv2d, nn.BatchNorm2d), (nn.Conv3d, nn.BatchNorm3d))



def fold_batch_norms(model: nn.Module) -> nn.Module:
    '''
    #* モデルの写しを eval mode にし、各 BatchNorm をそのすぐ前の conv に畳み込んで返す。
    #! 同じ親の module の中で隣り合う conv と BatchNorm だけを畳み込む。
    '''
    
    # 写しの中の、すべての module の子を見る
    folded = copy.deepcopy(model).eval()
    for module in list(folded.modules()):
        children = list(module.named_children())
        
        # 各組 (conv, BatchNorm) を、畳み込んだ conv と Identity に置き換える
        for (conv_name, conv), (norm_name, norm) in zip(children, children[1:]):
            for conv_type, norm_type in FOLD_PAIRS:
                if isinstance(conv, conv_type) and isinstance(norm, norm_type):
                    setattr(module, conv_name, fuse_conv_bn_eval(conv, norm))
                    setattr(module, norm_name, nn.Identity()                )
    return folded
