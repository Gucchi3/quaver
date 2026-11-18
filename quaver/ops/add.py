# @file add.py
# @brief 量子化した 2 つのテンソルの残差の足し算
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import device_define
from ..scales    import QuantParams, params_attach, params_codes, params_math_dtype, params_named, params_note, params_round
from .requantize import build_requantize, requantize_params, requantize_rescale
from .types      import FixedPoint



def add_fake(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* training mode の残差の足し算: add_integer と同じ整数の計算で、勾配はそのまま通す。
    #? 学習する出力の scale（LSQ）にも勾配が来る。計算には params_math_dtype() の型を使う。
    '''
    
    dtype         = params_math_dtype()
    fixed         = [requantize_params(params.code_scale / out_params.code_scale, style_name) for params in (a_params, b_params)]
    codes, inside = _add_codes(params_codes(a, a_params, dtype), a_params.zero_point, fixed[0], params_codes(b, b_params, dtype), b_params.zero_point, fixed[1], out_params, dtype)
    return params_attach(a + b, codes, out_params, inside, dtype)


def add_integer(q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* integer mode の残差の足し算: 両方の入力を丸めずに出力の code の単位に直し、足して、和を 1 回だけ丸める。
    #? デバイスでは、各入力の (code - zero point) x M0 を大きい方の shift n にそろえ、2 つを足し、和を n で 1 回だけ丸める。TFLite と CMSIS-NN も、和のあとでだけ丸める。
    #? デバイスの op torch.ops.quaver.add がこれをする。各入力の静的な M0 と n はここで求める。
    #! 計算はいつも float64。|和| x 2^n < 2^53 なら和は正確である: 出力の範囲の中の和なら n = 40（INT8）まで成り立ち、それより大きな和はどのみち切り捨てられる。
    '''
    
    fixed = [requantize_params(params.code_scale / out_params.code_scale, style_name) for params in (a_params, b_params)]
    codes = torch.ops.quaver.add(q_a, a_params.zero_point.to(torch.int64), fixed[0].multiplier, fixed[0].shift, q_b, b_params.zero_point.to(torch.int64), fixed[1].multiplier, fixed[1].shift, out_params.zero_point.to(torch.int64),
                               out_params.number_format.name, style_name)
    return params_note(codes, out_params)


@device_define("add")
def add_device(q_a: torch.Tensor, a_zero_point: torch.Tensor, a_multiplier: torch.Tensor, a_shift: torch.Tensor, q_b: torch.Tensor, b_zero_point: torch.Tensor, b_multiplier: torch.Tensor, b_shift: torch.Tensor, out_zero_point: torch.Tensor,
               out_format: str, style_name: str) -> torch.Tensor:
    '''
    #* デバイスの op "add": 各入力の (code - zero point) x M0 / 2^n を丸めずに足し、out_format に 1 回だけ丸めて、出力の zero point を足す。
    #? add_integer() が torch.ops.quaver.add として呼ぶ。丸めは style のもの（FP の出力は half to even に丸める）。
    '''
    
    style    = build_requantize(style_name)
    codes, _ = _add_codes(q_a, a_zero_point, FixedPoint(a_multiplier, a_shift, style), q_b, b_zero_point, FixedPoint(b_multiplier, b_shift, style), params_named(out_zero_point, out_format), torch.float64)
    return codes


def add_float(a: torch.Tensor, a_params: QuantParams, b: torch.Tensor, b_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* float のモード（calibration）: requantize せずに、float の和 a + b を返す。
    '''
    
    return a + b


def _add_codes(q_a: torch.Tensor, a_zero: torch.Tensor, a_fixed: FixedPoint, q_b: torch.Tensor, b_zero: torch.Tensor, b_fixed: FixedPoint, out_params: QuantParams, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* 各入力を出力の code の単位に正確に直し、2 つを足して、和を出力の形式に 1 回だけ丸め、範囲の中にあった code を返す。
    #? add_fake() と add_device() が使うので、2 つのモードは一致する（float64 ではいつも、float32 でもほとんどいつも）。
    #! out_params は、zero point と形式だけを使う。
    '''
    
    # 各入力（zero point を引いたもの）を、丸めずに出力の code の単位に直す
    parts = [requantize_rescale(codes - zero.to(torch.int64), fixed, dtype) for codes, zero, fixed in ((q_a, a_zero, a_fixed), (q_b, b_zero, b_fixed))]
    
    # 正確な 2 つの部分を足し、和を出力の形式に 1 回だけ丸める（style の丸め。FP の出力は half even）
    return params_round(parts[0] + parts[1], out_params, a_fixed.style.round_mode)
