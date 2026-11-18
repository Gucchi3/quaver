# @file accumulate.py
# @brief conv、linear、matmul の正確な累積（INT32、ブロックごとの scale のテンソルではもっと広い）
# @author Kohei Yamaguchi, Claude

from collections.abc import Callable
from torch.nn        import functional as F

import torch

from ..formats import NumberFormat, device_define, integer_format
from ..scales  import QuantParams, params_bits, params_integers, params_named, params_unit

# INT32 の累積。あふれは再現しない: 和は正確（int64）で、デバイスは INT32 の中に収まると考える
ACCUMULATOR_FORMAT = integer_format(32, signed=True)

# ブロックごとの scale のテンソルの累積: 和の前にすべてのブロックを 1 つの刻みにそろえるので、32 bit より多く要る（NVFP4 で K = 1024 なら 54 bit）
WIDE_ACCUMULATOR_FORMAT = integer_format(62, signed=True)

# float64 の conv や matmul 1 回は、和が 2^EXACT_BITS より小さい間は正確である（どの計算の仕方でも、誤差は 0.5 よりずっと小さい）
EXACT_BITS = 46



def accumulate_scale(x_params: QuantParams, w_params: QuantParams) -> torch.Tensor:
    '''
    #* conv か linear の層の累積の scale を、出力チャネルごとに 1 つ返す（平らにして）。
    #? 累積の scale = 入力の整数の刻み x 重みの整数の刻み（params_unit: code の scale。ブロックごとの scale のテンソルなら、すべてのブロックに共通の刻み）。
    #! 入力の刻みはテンソルで 1 つでなければならない。重みの刻みは、テンソルで 1 つか、出力チャネルごと（軸 0）に 1 つにする。
    '''
    
    return params_unit(x_params).reshape(-1) * params_unit(w_params).reshape(-1)


def accumulate_format(x_params: QuantParams, w_params: QuantParams) -> NumberFormat:
    '''
    #* 累積の形式を返す: INT32 か、和にブロックごとの scale のテンソルが入るときは wide の形式。
    #! 計算はこれを使い続ける。model_export() は、accumulate_bits() が 32 以下なら、wide の形式の代わりに int32 と書く。
    '''
    
    return WIDE_ACCUMULATOR_FORMAT if x_params.block is not None or w_params.block is not None else ACCUMULATOR_FORMAT


def accumulate_bits(x_params: QuantParams, w_params: QuantParams, count: int, bias_peak: int = 0) -> int:
    '''
    #* count 個の積とバイアスの和に、一番悪い場合に要るビット数（符号を含む）を、形式と |バイアスの code| の最大から求めて返す。
    #? バイアスなし: 両側の params_bits() + ceil(log2(count)) + 1。バイアスは、和をそれより大きくしうるときだけ、もう 1 bit 足す。model_export() が使う。
    '''
    
    largest = 2 ** (params_bits(x_params) + params_bits(w_params) + (count - 1).bit_length()) - 1
    return (largest + bias_peak).bit_length() + 1


def accumulate_conv2d(q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams, stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, dilation: int | tuple[int, int] = 1, groups: int = 1) -> torch.Tensor:
    '''
    #* 整数の code での conv の、正確な累積（int64）を返す。
    #? 先に zero point を引くので、パディングは正確な 0 を足す。ブロックごとの scale の code には、先にそのブロックの scale の code をかける。デバイスの op torch.ops.quaver.accumulate_conv2d が計算する。
    '''
    
    x, w  = _accumulate_parts(x_params), _accumulate_parts(w_params)
    pairs = [list(value) if isinstance(value, tuple) else [value, value] for value in (stride, padding, dilation)]
    return torch.ops.quaver.accumulate_conv2d(q_x, x[0], x[1], q_w, w[0], w[1], x[2], x[3], w[2], w[3], accumulate_format(x_params, w_params).name, pairs[0], pairs[1], pairs[2], groups)


def accumulate_linear(q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
    '''
    #* 整数の code での linear の層の、正確な累積（int64）を返す。
    #? デバイスの op torch.ops.quaver.accumulate_linear が計算する。
    '''
    
    x, w = _accumulate_parts(x_params), _accumulate_parts(w_params)
    return torch.ops.quaver.accumulate_linear(q_x, x[0], x[1], q_w, w[0], w[1], x[2], x[3], w[2], w[3], accumulate_format(x_params, w_params).name)


def accumulate_matmul(q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams) -> torch.Tensor:
    '''
    #* 整数の code の行列積の、正確な累積（int64）を返す。
    #? デバイスの op torch.ops.quaver.accumulate_matmul が計算する。
    '''
    
    a, b = _accumulate_parts(a_params), _accumulate_parts(b_params)
    return torch.ops.quaver.accumulate_matmul(q_a, a[0], a[1], q_b, b[0], b[1], a[2], a[3], b[2], b[3], accumulate_format(a_params, b_params).name)


@device_define("accumulate_conv2d")
def accumulate_conv2d_device(q_x: torch.Tensor, x_zero_point: torch.Tensor, x_block: torch.Tensor | None, q_w: torch.Tensor, w_zero_point: torch.Tensor, w_block: torch.Tensor | None, x_format: str, x_scale_format: str, w_format: str,
                             w_scale_format: str, out_format: str, stride: list[int], padding: list[int], dilation: list[int], groups: int) -> torch.Tensor:
    '''
    #* デバイスの op "accumulate_conv2d": conv の (code - zero point) x （ブロックの scale の code）の積の正確な和を、int64 で返す。
    #? x_block と w_block は各値のブロックの scale の code（ブロックのないテンソルでは None）で、scale の形式がその名前を表す（ブロックがないときは ""）。
    #! out_format はデバイスでの累積の名前: int32、または和にブロックごとの scale のテンソルが入るときは int62。和そのものは正確（int64）である。
    '''
    
    x_params = params_named(x_zero_point, x_format, x_block, x_scale_format)
    w_params = params_named(w_zero_point, w_format, w_block, w_scale_format)
    return _accumulate_exact(lambda x, w: F.conv2d(x, w, None, stride, padding, dilation, groups), q_x, x_params, q_w, w_params, q_w[0].numel())


@device_define("accumulate_linear")
def accumulate_linear_device(q_x: torch.Tensor, x_zero_point: torch.Tensor, x_block: torch.Tensor | None, q_w: torch.Tensor, w_zero_point: torch.Tensor, w_block: torch.Tensor | None, x_format: str, x_scale_format: str, w_format: str,
                             w_scale_format: str, out_format: str) -> torch.Tensor:
    '''
    #* デバイスの op "accumulate_linear": accumulate_conv2d_device() と同じように、linear の層の積の正確な和を int64 で返す。
    '''
    
    x_params = params_named(x_zero_point, x_format, x_block, x_scale_format)
    w_params = params_named(w_zero_point, w_format, w_block, w_scale_format)
    return _accumulate_exact(F.linear, q_x, x_params, q_w, w_params, q_w.shape[-1])


@device_define("accumulate_matmul")
def accumulate_matmul_device(q_a: torch.Tensor, a_zero_point: torch.Tensor, a_block: torch.Tensor | None, q_b: torch.Tensor, b_zero_point: torch.Tensor, b_block: torch.Tensor | None, a_format: str, a_scale_format: str, b_format: str,
                             b_scale_format: str, out_format: str) -> torch.Tensor:
    '''
    #* デバイスの op "accumulate_matmul": accumulate_conv2d_device() と同じように、行列積 a @ b の積の正確な和を int64 で返す。
    '''
    
    a_params = params_named(a_zero_point, a_format, a_block, a_scale_format)
    b_params = params_named(b_zero_point, b_format, b_block, b_scale_format)
    return _accumulate_exact(accumulate_product, q_a, a_params, q_b, b_params, q_a.shape[-1])


def accumulate_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''
    #* a @ b を返す。大きさの 1 つが 1 のときは、代わりに broadcast の積と和を使う。
    #! PyTorch の一部の CUDA のビルド（2.13 で見た）は、大きさ 1 を含む積を、初回に C をコンパイルする Triton のカーネルに送る。
    '''
    
    if 1 in (a.shape[-2], a.shape[-1], b.shape[-1]):
        return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(dim=-2)
    return torch.matmul(a, b)


def _accumulate_parts(params: QuantParams) -> tuple[torch.Tensor, torch.Tensor | None, str, str]:
    '''
    #* パラメータから、デバイスの op が要るものを返す: zero point（int64）、ブロックの scale の code（または None）、形式の名前、scale の形式の名前（ブロックがないときは ""）。
    #? accumulate_conv2d()、accumulate_linear()、accumulate_matmul() が、両側に使う。
    '''
    
    if params.block is None:
        return params.zero_point.to(torch.int64), None, params.number_format.name, ""
    return params.zero_point.to(torch.int64), params.block.codes, params.number_format.name, params.block.scale_format.name


def _accumulate_exact(op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], q_a: torch.Tensor, a_params: QuantParams, q_b: torch.Tensor, b_params: QuantParams, size: int) -> torch.Tensor:
    '''
    #* 両側の整数で op（size 個の積を足す conv、linear、matmul）を正確に計算し、int64 で返す。
    #? デバイスの op が使う。float64 の op 1 回で 2^EXACT_BITS を超えうるときは、b を少ないビットの部分に分け、部分ごとの結果をその桁に足す。
    #! 幅は形式から求める（params_bits）ので、GPU から何も読まない。62 bit より多く要る和は ValueError になる。
    '''
    
    # 両側の整数と、和と b の 1 つの部分の幅
    a, b       = params_integers(q_a, a_params), params_integers(q_b, b_params)
    bits_a     = params_bits(a_params)
    bits_b     = params_bits(b_params)
    bits_sum   = size.bit_length()
    part       = EXACT_BITS - bits_a - bits_sum
    if bits_a + bits_b + bits_sum > 62 or part < 1:
        raise ValueError(f"The exact sum needs {bits_a} + {bits_b} + {bits_sum} bits, more than int64 and float64 parts keep. Use narrower formats or fewer values in one sum.")
    
    # b の各部分（下の部分は 0〜2^part - 1、一番上の部分は符号を持つ）を、それぞれ正確な float64 の op 1 回で計算し、その桁に足す
    count = -(-bits_b // part)
    total = torch.round(op(a.double(), (b if count == 1 else b & (2**part - 1)).double())).to(torch.int64)
    for index in range(1, count):
        digit = b >> (index * part) if index == count - 1 else (b >> (index * part)) & (2**part - 1)
        total = total + (torch.round(op(a.double(), digit.double())).to(torch.int64) << (index * part))
    return total
