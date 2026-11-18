# @file sync.py
# @brief GPU から CPU への同期を数える
# @author Kohei Yamaguchi, Claude

from collections.abc import Callable

import torch
import warnings

SYNC_MESSAGE = "called a synchronizing CUDA operation"



def sync_count(fn: Callable, *args: object) -> int:
    '''
    #* fn(*args) を 1 回動かし、それが起こす GPU から CPU への同期を数える。
    #? 自分で書いた層に同期がないかを確かめるのに使う。CUDA graph には同期が 0 個であることが要る。
    #! GPU がないときは 0 を返す。
    '''
    
    # GPU がないと数えるものがない
    if not torch.cuda.is_available():
        return 0
    
    # PyTorch が同期のたびに警告を出すようにして、fn を動かす
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn(*args)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum(1 for warning in caught if SYNC_MESSAGE in str(warning.message))
