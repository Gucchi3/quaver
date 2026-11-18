# @file requantize.py
# @brief multiplier と shift で、累積の scale を出力の code に直す
# @author Kohei Yamaguchi, Claude

##########################################################
'''
requantize の style を足すとき
1. デバイスに合わせて RequantizeStyle(name, multiplier_max, shift_max, round_mode) を作る。
2. REQUANTIZE_REGISTRY に "名前: style" を足す。
3. 名前を style_name として渡す。
'''
##########################################################

import torch

from ..formats import RoundMode, build_format, device_define, round_half_away
from ..scales  import QuantParams, params_attach, params_codes, params_math_dtype, params_note, params_round
from .types    import FixedPoint, RequantizeStyle
from .wide     import wide_round_format

REQUANTIZE_REGISTRY: dict[str, RequantizeStyle] = {
    "mul16_shift": RequantizeStyle("mul16_shift", multiplier_max=2**16 - 1, shift_max=94, round_mode="half_up"  ),
    "shift_only" : RequantizeStyle("shift_only" , multiplier_max=1        , shift_max=94, round_mode="half_up"  ),
    "exact"      : RequantizeStyle("exact"      , multiplier_max=0        , shift_max=0 , round_mode="half_away"),
}



def build_requantize(name: str) -> RequantizeStyle:
    '''
    #* 名前の requantize の style を返す。
    #? mul16_shift: 16 bit の multiplier と (acc x M0 + 2^(n-1)) >> n。shift_only: 2 のべき乗の scale（M0 = 1）。exact: 実数の比そのもの。
    #! 新しい style は REQUANTIZE_REGISTRY に足す。
    '''
    
    if name not in REQUANTIZE_REGISTRY:
        raise KeyError(f"Unknown requantize style {name!r}. Available: {', '.join(REQUANTIZE_REGISTRY)}")
    return REQUANTIZE_REGISTRY[name]


def requantize_params(ratio: torch.Tensor, style_name: str) -> FixedPoint:
    '''
    #* 比を表す multiplier M0 と shift n（ratio = M0 / 2^n）を、1 つの式で求める（shift についてのループはない）。
    #? n は、M0 = round(ratio x 2^n) が multiplier_max に収まる一番大きな shift なので、M0 はすべてのビットを使う（mul16_shift では 16 bit）。n < 0 は左シフト（2^16 以上の比）で、-30 まで。
    #! 静的な scale だけに使う: デバイスは M0 と n を Python から受け取る。データで変わる比には requantize_ratio() を使う。こちらは整数しか使わない。
    #! op torch.ops.quaver.fixed_point が求めるので、書き出したモデルはそれを定数に畳み込む（scale を毎回測るときは、ノードを 1 つ残す）。
    '''
    
    # exact の style は比そのものを持つ
    style = build_requantize(style_name)
    ratio = ratio.detach().double()
    if style.multiplier_max == 0:
        return FixedPoint(ratio, torch.zeros_like(ratio, dtype=torch.int64), style)
    
    # デバイスの op で求めた multiplier と shift
    multiplier, shift = torch.ops.quaver.fixed_point(ratio, style_name)
    return FixedPoint(multiplier, shift, style)


@device_define("fixed_point")
def requantize_fixed_device(ratio: torch.Tensor, style_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* op "fixed_point": 比の multiplier M0 と shift n（ratio = M0 / 2^n）。M0 を multiplier_max 以下に保つ一番大きな n を使う。
    #? requantize_params() が、multiplier を持つ style のときに torch.ops.quaver.fixed_point として呼ぶ。ratio は float64。
    '''
    
    style       = build_requantize(style_name)
    _, exponent = torch.frexp(ratio)
    shift       = style.multiplier_max.bit_length() - exponent.to(torch.int64)
    shift       = torch.where(_requantize_multiplier(ratio, shift) > style.multiplier_max, shift - 1, shift).clamp(-30, style.shift_max)
    multiplier  = _requantize_multiplier(ratio, shift).clamp(0, style.multiplier_max)
    return multiplier.to(torch.int64), shift


def requantize_ratio(numerator: torch.Tensor, denominator: torch.Tensor, style_name: str) -> FixedPoint:
    '''
    #* numerator / denominator の M0 と n を、デバイスが動きながらするのと同じく、整数だけ（shift、かけ算 1 回、整数の割り算）で求める。
    #? NEQ と attention のスコアが、データで変わる比に使う。M0 = round(numerator x 2^n / denominator)（ちょうど半分は上へ）で、n は収まる一番大きな値（n < 0 は左シフト）。
    #! 1 <= numerator < 2^62、1 <= denominator < 2^46 の int64 のテンソルを渡すので、どの手順も 2^63 より小さい。multiplier を持つ style だけに使う（exact には使わない）。
    #! float64 で 2 のべき乗に切り上がる numerator では、frexp はビットを 1 つ多く数える。そのとき M0 は 1 shift 下から始まって手順が 1 回少なくて済むので、M0 と n は変わらない。
    '''
    
    # この入口で確かめる: exact の style は実数の比を持つので、デバイスは動きながらそれを求められない
    style = build_requantize(style_name)
    if style.multiplier_max == 0:
        raise ValueError(f"The style {style_name!r} keeps a real ratio, but this ratio changes with the data and needs M0 and n. Use a style with a multiplier (mul16_shift or shift_only).")
    
    # M0 を 2^bits のすぐ下にする shift を、ビットの長さから求める（本当の比とは最大で 2 倍ずれうるので、M0 は最大で 2^(bits + 1)）
    lengths    = [torch.frexp(values.double())[1].to(torch.int64) for values in (numerator, denominator)]
    shift      = style.multiplier_max.bit_length() - (lengths[0] - lengths[1])
    multiplier = _requantize_divide(numerator, denominator, shift)
    
    # M0 が大きすぎる間、shift を 1 つずつ下げる。1 回ごとに M0 が半分になるので、2 回で必ず足りる
    for _ in range(2):
        shift      = torch.where(multiplier > style.multiplier_max, shift - 1, shift)
        multiplier = _requantize_divide(numerator, denominator, shift)
    return FixedPoint(multiplier, shift, style)


def requantize_rescale(acc: torch.Tensor, fixed_point: FixedPoint, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    '''
    #* 累積に比をかける: 丸める前の acc x M0 / 2^n を、float の型 dtype で求める。
    #? float64 では、32 bit の累積と 16 bit の multiplier の積は 2^53 より小さいので、結果は正確である。
    #! float32 では、積は 24 bit しか持たないので、2 つの code のちょうど中間にとても近い値は、もう一方の code に丸まることがある。
    '''
    
    if fixed_point.style.multiplier_max == 0:
        return acc.to(dtype) * fixed_point.multiplier.to(dtype)
    return torch.ldexp(acc.to(dtype) * fixed_point.multiplier.to(dtype), -fixed_point.shift)


def requantize_fake(x: torch.Tensor, x_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* training mode: デバイスと同じ方法で requantize し、勾配はそのまま通す（学習する出力の scale には LSQ の勾配）。
    #? x から code を取り戻し、requantize_integer の整数の計算を params_math_dtype() の型でする（precision_set_math() で変えていなければ float64）。
    #! x は code の正確な値（累積か、ほかの op の出力）でなければならないので、TF32 を止めておく。
    '''
    
    dtype         = params_math_dtype()
    codes, inside = _requantize_codes(params_codes(x, x_params, dtype), x_params, out_params, style_name, dtype)
    return params_attach(x, codes, out_params, inside, dtype)


def requantize_integer(acc: torch.Tensor, acc_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* integer mode: デバイスと同じく、累積（またはどんな code でも）を出力の code に直す。
    #? 比は acc_params.code_scale / out_params.code_scale で、multiplier と shift にする。積は int64 の計算だけで丸める（wide_round_format）ので、どんな累積でも正確である。
    #! 先に acc_params の zero point を引くので、非対称の形式の code も実数の値を保つ。累積の zero point は 0。exact の style は float64 を使う。
    '''
    
    # exact の style は実数の比を持つので、training mode の float64 の計算を動かす
    fixed_point = requantize_params(acc_params.code_scale / out_params.code_scale, style_name)
    if fixed_point.style.multiplier_max == 0:
        codes, _ = _requantize_codes(acc, acc_params, out_params, style_name, torch.float64)
        return codes
    
    # multiplier と shift: デバイスの op。M0 と n はここで求める
    codes = torch.ops.quaver.requantize(acc, acc_params.zero_point.to(torch.int64), fixed_point.multiplier, fixed_point.shift, out_params.zero_point.to(torch.int64), out_params.number_format.name, fixed_point.style.round_mode)
    return params_note(codes, out_params)


@device_define("requantize")
def requantize_device(acc: torch.Tensor, zero_point: torch.Tensor, multiplier: torch.Tensor, shift: torch.Tensor, out_zero_point: torch.Tensor, out_format: str, mode: RoundMode) -> torch.Tensor:
    '''
    #* デバイスの op "requantize": (acc - zero_point) x M0 / 2^n を out_format に 1 回だけ丸め、出力の zero point を足して、範囲の両端で止める。
    #? requantize_integer() が torch.ops.quaver.requantize として呼ぶ。M0 と n は静的（requantize_params() が求め、デバイスはそれを受け取る）。
    #! int64 の計算だけを使う（wide_round_format）ので、acc がどの幅でも正確。ここに来るのは multiplier を持つ style だけ。
    '''
    
    centered = acc - zero_point
    return wide_round_format(centered, multiplier, torch.ones_like(centered), shift, build_format(out_format), mode, out_zero_point)


def requantize_float(x: torch.Tensor, x_params: QuantParams, out_params: QuantParams, style_name: str) -> torch.Tensor:
    '''
    #* float のモード（calibration）: requantize せずに、x をそのまま返す。
    #? 層はこの前に出力の範囲を測る（Quantizer.measure()）ので、PyTorch の observer と同じく、scaler は float の値を見る。
    '''
    
    return x


def _requantize_codes(acc: torch.Tensor, acc_params: QuantParams, out_params: QuantParams, style_name: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* 入力の zero point を引いて出力の code に直し、範囲の中にあった code を返す。
    #? requantize_fake() と requantize_integer() が使うので、2 つのモードは一致する（float64 ではいつも、float32 でもほとんどいつも）。
    #! multiplier と shift はいつも float64 で求め、各値の計算だけに dtype を使う。
    '''
    
    fixed_point = requantize_params(acc_params.code_scale / out_params.code_scale, style_name)
    centered    = acc - acc_params.zero_point.to(torch.int64)
    return params_round(requantize_rescale(centered, fixed_point, dtype), out_params, fixed_point.style.round_mode)


def _requantize_multiplier(ratio: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    '''
    #* ratio x 2^shift を丸めて multiplier にする。ちょうど半分は上へ丸める（比は正）。
    #? requantize_params() が 2 回使う。
    '''
    
    return round_half_away(torch.ldexp(ratio, shift))


def _requantize_divide(numerator: torch.Tensor, denominator: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    '''
    #* 整数の割り算で round(numerator x 2^shift / denominator) を返す（ちょうど半分は上へ）。shift が負のときは、代わりに denominator を上へずらす。
    #? requantize_ratio() が 3 回使う。
    #! 余りで丸めを決めるので、requantize_ratio() が許す大きさを超えて 2 倍にする所はない。
    '''
    
    up    = torch.bitwise_left_shift(numerator  , shift.clamp_min(0)   )
    down  = torch.bitwise_left_shift(denominator, (-shift).clamp_min(0))
    whole = torch.div(up, down, rounding_mode="floor")
    return whole + (2 * (up - whole * down) >= down).to(torch.int64)
