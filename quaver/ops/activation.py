# @file activation.py
# @brief 整数の code での ReLU と ReLU6
# @author Kohei Yamaguchi, Claude

import torch

from ..formats import device_define, ste_attach
from ..scales  import QuantParams, params_codes, params_math_dtype, params_note, params_values
from .types    import ActivationKind



def activation_fake(x: torch.Tensor, x_params: QuantParams, kind: ActivationKind) -> torch.Tensor:
    '''
    #* training mode の ReLU か ReLU6: activation_integer と同じく code を clamp し、勾配はそのまま通す。
    #? kind は "none"、"relu"、"relu6" のどれか。計算には params_math_dtype() の型を使う。
    '''
    
    dtype         = params_math_dtype()
    zero          = x_params.zero_point.to(torch.int64)
    six           = zero + torch.round(6.0 / x_params.code_scale.double()).to(torch.int64) if kind == "relu6" else zero
    codes, inside = _activation_codes(params_codes(x, x_params, dtype), zero, six, kind)
    return ste_attach(x, params_values(codes, x_params, dtype), inside)


def activation_integer(q_x: torch.Tensor, x_params: QuantParams, kind: ActivationKind) -> torch.Tensor:
    '''
    #* integer mode の ReLU か ReLU6: code を、0 の code と 6 の code の間に clamp する。
    #? デバイスの op torch.ops.quaver.activation が clamp する。6 の code は静的（ここで求めてデバイスに渡す）。活性化なしのときは、op を使わずに code をそのまま返す。
    #? どちらの場合も code にパラメータの印を付けるので、書き出したモデルは累積の scale が分かる。
    #! requantize の前の累積（INT32）に使う。そうすると、上限と下限が正確な整数になる。
    '''
    
    if kind == "none":
        return params_note(q_x, x_params)
    zero = x_params.zero_point.to(torch.int64)
    six  = zero + torch.round(6.0 / x_params.code_scale.double()).to(torch.int64) if kind == "relu6" else zero
    return params_note(torch.ops.quaver.activation(q_x, zero, six, kind), x_params)


@device_define("activation")
def activation_device(q_x: torch.Tensor, zero_point: torch.Tensor, six: torch.Tensor, kind: ActivationKind) -> torch.Tensor:
    '''
    #* デバイスの op "activation": ReLU は code を 0 の code（zero point）で、ReLU6 はさらに 6 の code（six）で clamp する。
    #? activation_integer() が、"relu" と "relu6" のときに torch.ops.quaver.activation として呼ぶ。ReLU では six に zero point を渡し、six は使わない。
    '''
    
    codes, _ = _activation_codes(q_x, zero_point, six, kind)
    return codes


def activation_float(x: torch.Tensor, x_params: QuantParams, kind: ActivationKind) -> torch.Tensor:
    '''
    #* float のモード（calibration）: float の値に ReLU か ReLU6 をかける。
    '''
    
    if kind == "none":
        return x
    return x.clamp(0.0, 6.0 if kind == "relu6" else None)


def _activation_codes(codes: torch.Tensor, zero: torch.Tensor, six: torch.Tensor, kind: ActivationKind) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* code を 0 の code で（ReLU6 では 6 の code でも）clamp し、変わらなかった code を返す。
    #? activation_fake() と activation_device() が使うので、2 つのモードはいつも一致する。
    '''
    
    # 活性化なしのときは、すべての code をそのまま残す
    if kind == "none":
        return codes, torch.ones_like(codes, dtype=torch.bool)
    
    # 0 の code で clamp し、ReLU6 では 6 の code でも clamp する
    limited = torch.maximum(codes, zero)
    if kind == "relu6":
        limited = torch.minimum(limited, six)
    return limited, limited == codes
