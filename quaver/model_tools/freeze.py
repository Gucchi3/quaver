# @file freeze.py
# @brief scaler の範囲の更新を止める、またはもう一度始める
# @author Kohei Yamaguchi, Claude

from torch import nn

from ..scales import RangeScaler, SharedScaler
from .collect import collect_scalers



def freeze_scalers(model: nn.Module, frozen: bool = True, step: int | None = None) -> int:
    '''
    #* max、percentile、共有の scaler の範囲の更新を止める（frozen=True）か、もう一度始める（frozen=False）。step を渡すと、あとで自動で止まるようにする。設定した scaler の数を返す。
    #? step なしのときは、数エポックのあとで呼ぶ（E_AI_deit の range_freeze_epoch）。step ありのときは、学習の前に 1 回、step = エポック数 x 1 エポックの step 数 で呼ぶ。
    #! frozen を変えたら、CUDA graph を記録し直す。記録の前に決めた step なら、止める処理はデバイスの上で起きるので、記録し直さなくてよい。
    '''
    
    scalers = [*collect_scalers(model, RangeScaler), *collect_scalers(model, SharedScaler)]
    for _, scaler in scalers:
        if step is None:
            scaler.frozen = frozen
        else:
            scaler.freeze_step = step
    return len(scalers)
