# @file normalize.py
# @brief NEQ: トークンごとかテンソルで（SimA）L1 正規化と量子化を 1 度にし、その積を requantize する
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import NumberFormat, build_format, device_define, ste_attach
from ..scales    import QuantParams, params_attach, params_codes, params_math_dtype, params_named, params_note, params_round, params_values
from .accumulate import ACCUMULATOR_FORMAT
from .requantize import build_requantize, requantize_params, requantize_ratio, requantize_rescale
from .types      import FixedPoint, NeqScope

NORM_MIN          = torch.finfo(torch.float32).eps
NORM_BITS         = 20
NORM_INVERSE_BITS = 24



def normalize_l1_fake(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, style_name: str, scope: NeqScope = "tensor") -> tuple[torch.Tensor, QuantParams]:
    '''
    #* training mode の NEQ: トークン（最後の軸）ごとに x / L1(x) を求め、割り算を scale に入れて number_format に量子化する。
    #? normalize_l1_integer と同じ整数の計算を、params_math_dtype() の型でする。勾配は float の正規化に従う。
    #! x はたいてい conv の累積（fake の値）なので、code を正確に取り戻すために TF32 を止めておく。
    '''
    
    dtype       = params_math_dtype()
    align       = requantize_params(x_params.code_scale / x_params.scale.amax(), style_name)
    codes, peak = _normalize_l1_codes(params_codes(x, x_params, dtype), x_params.zero_point, align, number_format, style_name, scope, dtype)
    scale       = (peak.double() / (2**NORM_BITS * number_format.max_value)).float()
    params      = QuantParams(scale, torch.zeros([1] * peak.ndim, device=peak.device), number_format, peak=peak)
    reference   = x / x.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
    return ste_attach(reference, params_values(codes, params, dtype)), params


def normalize_l1_integer(q_x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, style_name: str, scope: NeqScope = "tensor") -> tuple[torch.Tensor, QuantParams]:
    '''
    #* integer mode の NEQ（Normalization-Equivalent Quantization）: 各トークン（最後の軸）を L1 正規化して量子化する処理を 1 度にし、1 / L1 を scale に入れる。
    #? scope "tensor" は論文の Algorithm 2（実機で使う設定）: サンプルと head ごとに scale を 1 つ、max|v| / L1 が一番大きなトークンから決める。"token": トークンごとに scale を 1 つ持つ。
    #? デバイスの op torch.ops.quaver.normalize_l1 が計算する。チャネルを一番大きなチャネルの scale にそろえる静的な M0 と n は、ここで求める。
    #? パラメータは整数 p（peak）を持ち、attention map がそれを使う。scale は p / (2^NORM_BITS x 形式の一番大きな値)。
    #! 整数だけを使う: p = floor(max|v| x 2^NORM_BITS / L1)（"tensor" ではトークンでの最大）、M0 = round(top x 2^NORM_BITS x 2^n / (L1 x p))、code = (v x M0 + 2^(n-1)) >> n。
    '''
    
    align       = requantize_params(x_params.code_scale / x_params.scale.amax(), style_name)
    codes, peak = torch.ops.quaver.normalize_l1(q_x, x_params.zero_point.to(torch.int64), align.multiplier, align.shift, number_format.name, style_name, scope)
    scale       = (peak.double() / (2**NORM_BITS * number_format.max_value)).float()
    params      = QuantParams(scale, torch.zeros([1] * peak.ndim, device=peak.device), number_format, peak=peak)
    return params_note(codes, params), params


@device_define("normalize_l1")
def normalize_l1_device(q_x: torch.Tensor, zero_point: torch.Tensor, multiplier: torch.Tensor, shift: torch.Tensor, out_format: str, style_name: str, scope: NeqScope) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* デバイスの op "normalize_l1"（NEQ）: チャネルを 1 つの scale にそろえ（静的な M0 と n）、各トークンを整数で測り、NEQ の整数の multiplier で各値の scale を直して丸める。
    #? normalize_l1_integer() が torch.ops.quaver.normalize_l1 として呼ぶ。code と整数 p（int64、1〜2^NORM_BITS）を返す。p が scale を決める: p / (2^NORM_BITS x out_format の一番大きな値)。
    '''
    
    return _normalize_l1_codes(q_x, zero_point, FixedPoint(multiplier, shift, build_requantize(style_name)), build_format(out_format), style_name, scope, torch.float64)


def normalize_l1_float(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, style_name: str, scope: NeqScope = "tensor") -> tuple[torch.Tensor, QuantParams]:
    '''
    #* float のモード（calibration）: 各トークン（最後の軸）の float の L1 正規化と、NEQ がそれに与えるはずの scale（トークンごとか、テンソルで 1 つ）を返す。
    #? ほかのモードと同じく、scale は一番大きな |値| を形式の一番大きな値に合わせる。
    '''
    
    normalized = x / x.abs().sum(dim=-1, keepdim=True).clamp_min(NORM_MIN)
    largest    = normalized.abs().amax(dim=-1 if scope == "token" else (-2, -1), keepdim=True)
    scale      = (largest / number_format.max_value).clamp_min(NORM_MIN)
    return normalized, QuantParams(scale, torch.zeros_like(scale), number_format)


def normalize_l1_scores_fake(x: torch.Tensor, x_params: QuantParams, q_params: QuantParams, k_params: QuantParams, out_params: QuantParams, factor: float, style_name: str) -> torch.Tensor:
    '''
    #* training mode: 2 つの NEQ の出力の積 Q K^T（attention map）を normalize_l1_scores_integer と同じく requantize し、勾配はそのまま通す。
    #? x は行列積の fake の積（x_params）で、勾配の reference は x x factor。学習する出力の scale には LSQ の勾配が来る。
    '''
    
    # この入口で確かめる: Q と K は NEQ から来ていなければならない（NEQ が整数 p を与える）
    if q_params.peak is None or k_params.peak is None:
        raise ValueError("q_params and k_params must be the outputs of NEQ (normalize_l1), with their integer p (peak).")
    
    # 静的な部分 H と、デバイスの整数の計算
    dtype         = params_math_dtype()
    tops          = [params.number_format.max_value / params.number_format.step_min for params in (q_params, k_params)]
    static        = requantize_params(factor / (tops[0] * tops[1] * out_params.code_scale), style_name)
    codes, inside = _normalize_l1_scores_codes(params_codes(x, x_params, dtype), q_params.peak, k_params.peak, static, out_params, dtype)
    return params_attach(x * factor, codes, out_params, inside, dtype)


def normalize_l1_scores_integer(acc: torch.Tensor, x_params: QuantParams, q_params: QuantParams, k_params: QuantParams, out_params: QuantParams, factor: float, style_name: str) -> torch.Tensor:
    '''
    #* integer mode: 2 つの NEQ の出力の正確な積 Q K^T（attention map）を、デバイスが整数で求める multiplier で、出力の code に requantize する。
    #? 比は p_q x p_k x H: p_q と p_k は NEQ の整数の最大で、H = factor / (top_q x top_k x 出力の code の scale) は静的（M0 と n は Python で求める）。
    #? デバイスの op torch.ops.quaver.normalize_l1_scores が計算する。M0_H と n_H はここで求め、Q と K の整数 p（パラメータの peak）を使う。
    #! デバイスは p_q x p_k x M0_H を、整数で 16 bit の M0 と shift に切り詰める。SimA では factor は 1 / sqrt(key_dim)。multiplier は、サンプルと head ごとか、組ごとに 1 つ。
    '''
    
    # この入口で確かめる: Q と K は NEQ から来ていなければならない（NEQ が整数 p を与える）
    if q_params.peak is None or k_params.peak is None:
        raise ValueError("q_params and k_params must be the outputs of NEQ (normalize_l1), with their integer p (peak).")
    
    # 静的な部分 H と、デバイスの op
    tops   = [params.number_format.max_value / params.number_format.step_min for params in (q_params, k_params)]
    static = requantize_params(factor / (tops[0] * tops[1] * out_params.code_scale), style_name)
    codes  = torch.ops.quaver.normalize_l1_scores(acc, q_params.peak, k_params.peak, static.multiplier, static.shift, out_params.zero_point.to(torch.int64), out_params.number_format.name, style_name)
    return params_note(codes, out_params)


@device_define("normalize_l1_scores")
def normalize_l1_scores_device(acc: torch.Tensor, q_peak: torch.Tensor, k_peak: torch.Tensor, multiplier: torch.Tensor, shift: torch.Tensor, out_zero_point: torch.Tensor, out_format: str, style_name: str) -> torch.Tensor:
    '''
    #* デバイスの op "normalize_l1_scores": 2 つの NEQ の出力の Q K^T を requantize する。p_q x p_k x M0_H は整数で M0 と n に切り詰める。
    #? normalize_l1_scores_integer() が torch.ops.quaver.normalize_l1_scores として呼ぶ。M0_H と n_H（multiplier と shift）は静的。q_peak と k_peak は NEQ の整数 p（normalize_l1_device から）。
    '''
    
    static   = FixedPoint(multiplier, shift, build_requantize(style_name))
    codes, _ = _normalize_l1_scores_codes(acc, q_peak, k_peak, static, params_named(out_zero_point, out_format), torch.float64)
    return codes


def normalize_l1_scores_float(x: torch.Tensor, x_params: QuantParams, q_params: QuantParams, k_params: QuantParams, out_params: QuantParams, factor: float, style_name: str) -> torch.Tensor:
    '''
    #* float のモード（calibration）: requantize せずに、float の積 x factor を返す。
    '''
    
    return x * factor


def _normalize_l1_codes(q_x: torch.Tensor, zero_point: torch.Tensor, align: FixedPoint, number_format: NumberFormat, style_name: str, scope: NeqScope, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* チャネルを 1 つの scale にそろえ、各トークンを整数で測り、NEQ の整数の multiplier で各値の scale を直して丸める。code と整数 p を返す。
    #? normalize_l1_fake() と normalize_l1_device() が使うので、2 つのモードは一致する。align は各チャネルを一番大きなチャネルの scale にそろえる（静的）。ほかの multiplier はすべて整数の割り算で求める。dtype を使うのは、値の scale を直す所だけ。
    #! code の scale は p / (2^NORM_BITS x 形式の一番大きな値) で、p は 1〜2^NORM_BITS の整数。
    '''
    
    # INT32 への requantize と同じく、すべてのチャネルを一番大きなチャネルの scale にそろえる（静的な M0 と n）
    style      = build_requantize(style_name)
    aligned, _ = params_round(requantize_rescale(q_x - zero_point.to(torch.int64), align, dtype), params_named(torch.zeros((), dtype=torch.int64, device=q_x.device), ACCUMULATOR_FORMAT.name), style.round_mode)
    
    # 各トークンを測る: L1 ノルムと、正規化した最大 p = floor(max|v| x 2^NORM_BITS / L1)。テンソルで scale を 1 つにするときは、トークンでの最大を使う
    total = aligned.abs().sum(dim=-1, keepdim=True).clamp_min(1)
    peak  = torch.div(torch.bitwise_left_shift(aligned.abs().amax(dim=-1, keepdim=True), NORM_BITS), total, rounding_mode="floor")
    peak  = (peak if scope == "token" else peak.amax(dim=-2, keepdim=True)).clamp_min(1)
    
    # 各トークンの multiplier top x 2^NORM_BITS / (L1 x p) を、整数の割り算 2 回で求める: inverse = round(top x 2^(NORM_BITS + NORM_INVERSE_BITS) / p)、次に inverse / L1 を M0 と n にする
    top        = torch.full_like(peak, round(number_format.max_value / number_format.step_min) << (NORM_BITS + NORM_INVERSE_BITS))
    whole      = torch.div(top, peak, rounding_mode="floor")
    inverse    = whole + (2 * (top - whole * peak) >= peak).to(torch.int64)
    multiplier = requantize_ratio(inverse, total, style_name)
    fixed      = FixedPoint(multiplier.multiplier, multiplier.shift + NORM_INVERSE_BITS, style)
    
    # 各値の scale を直し、形式に丸める
    codes, _ = params_round(requantize_rescale(aligned, fixed, dtype), params_named(torch.zeros_like(peak), number_format.name), style.round_mode)
    return codes, peak


def _normalize_l1_scores_codes(acc: torch.Tensor, q_peak: torch.Tensor, k_peak: torch.Tensor, static: FixedPoint, out_params: QuantParams, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* 2 つの NEQ の出力の Q K^T を NEQ の整数の最大 p で requantize し、範囲の中にあった code を返す。
    #? normalize_l1_scores_fake() と normalize_l1_scores_device() が使うので、2 つのモードは一致する。static は H = factor / (top_q x top_k x 出力の code の scale) を M0 と n にしたもの（Python がデバイスに渡す）。
    #! 積 acc x M0 は |acc| < 2^37 の間は float64 で正確である。code が 8 bit まで、1 つの和の値が 2^20 個までなら成り立つ。out_params は、zero point と形式だけを使う。
    '''
    
    # デバイスは p_q x p_k x M0_H を整数で M0 と n に切り詰め、2^NORM_BITS の 2 つの scale は shift に入れる
    style = static.style
    pair  = requantize_ratio(q_peak * k_peak * static.multiplier, torch.ones_like(static.multiplier), style.name)
    fixed = FixedPoint(pair.multiplier, pair.shift + static.shift + 2 * NORM_BITS, style)
    return params_round(requantize_rescale(acc, fixed, dtype), out_params, style.round_mode)
