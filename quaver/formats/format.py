# @file format.py
# @brief 名前で選ぶ数値形式と、それに共通の丸め
# @author Kohei Yamaguchi, Claude

##########################################################
'''
数値形式を足すとき
1. integer_format() か fp_format() で形式を作る。
2. FORMAT_REGISTRY に "名前: 形式" を足す。
'''
##########################################################

import re
import torch

from .device  import device_define
from .fp      import fp_decode, fp_encode, fp_format, fp_step
from .integer import integer_decode, integer_encode, integer_format
from .round   import build_round
from .types   import NumberFormat, RoundMode

# 値の形式: int2〜int8、uint2〜uint8、fp4、ufp4。ブロックの scale（BlockScaler）の形式: ue4m3（NVFP4）、uint6、uint10（整数のブロックの scale）
FORMAT_REGISTRY: dict[str, NumberFormat] = {
    "int2"  : integer_format(2 , signed=True ),
    "int4"  : integer_format(4 , signed=True ),
    "int8"  : integer_format(8 , signed=True ),
    "uint2" : integer_format(2 , signed=False),
    "uint4" : integer_format(4 , signed=False),
    "uint6" : integer_format(6 , signed=False),
    "uint8" : integer_format(8 , signed=False),
    "uint10": integer_format(10, signed=False),
    "fp4"   : fp_format(2, 1, signed=True , nan_top=False),
    "ufp4"  : fp_format(2, 2, signed=False, nan_top=False),
    "ue4m3" : fp_format(4, 3, signed=False, nan_top=True , name="ue4m3"),
}



def build_format(name: str) -> NumberFormat:
    '''
    #* 名前の数値形式を返す。
    #? どの幅の INT の形式も名前で使える（累積の int32 と int62、uint1）ので、デバイスの op はどの形式も名前で受け取れる。
    #! 新しい形式は FORMAT_REGISTRY に足す。
    '''
    
    if name in FORMAT_REGISTRY:
        return FORMAT_REGISTRY[name]
    match = re.fullmatch(r"(u?)int([1-9][0-9]*)", name)
    if match is None:
        raise KeyError(f"Unknown number format {name!r}. Available: {', '.join(FORMAT_REGISTRY)}, or intN and uintN of any width")
    return integer_format(int(match.group(2)), signed=match.group(1) == "")


def format_round_mode(number_format: NumberFormat, mode: RoundMode | None = None) -> RoundMode:
    '''
    #* mode を返す。mode が None のときは、形式のいつもの丸めを返す: FP は half_even（IEEE、OCP MX、NVFP4 と同じ）、INT は half_away（TFLite と同じ）。
    #? Quantizer とサンプルが、標準の丸めを決めるのに使う。
    '''
    
    if mode is not None:
        return mode
    return "half_even" if number_format.kind == "fp" else "half_away"


def format_quantize(x: torch.Tensor, number_format: NumberFormat, mode: RoundMode, zero_point: torch.Tensor | None = None) -> torch.Tensor:
    '''
    #* x を形式の一番近い値に丸め、zero point を足して、範囲の両端で止める。
    #? INT と FP はこの同じコードを使う。違うのはグリッドの刻みだけで、INT は 1、FP は指数ごとの刻みになる。
    #! x は先に scale で割っておく。TFLite やデバイスと同じく、丸めは zero point を足す前にする。
    '''
    
    # 形式のグリッドに丸める
    round_fn = build_round(mode)
    if number_format.kind == "int":
        grid = round_fn(x)
    else:
        step = fp_step(x, number_format)
        grid = round_fn(x / step) * step
    
    # zero point を足し、範囲の両端で止める
    if zero_point is not None:
        grid = grid + zero_point
    return grid.clamp(number_format.min_value, number_format.max_value)


@device_define("quantize")
def format_codes(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, number_format: str, mode: RoundMode) -> torch.Tensor:
    '''
    #* デバイスの op "quantize": x / scale を形式に乗せた整数の code（int64）を、zero point を足して範囲の両端で止めて返す。
    #? Quantizer.encode() が torch.ops.quaver.quantize として呼ぶ。書き出したモデルでは入力を量子化する。重みは書き出しの前に code に畳み込む。
    #! FP の code は形式のグリッドの整数（value / step_min）なので、quaver のほかの整数の op と同じく、FP4 の 1.5 は 3 になる。
    '''
    
    named = build_format(number_format)
    return format_to_integer(format_quantize(x.float() / scale, named, mode, zero_point), named)


def format_encode(x: torch.Tensor, number_format: NumberFormat, mode: RoundMode) -> torch.Tensor:
    '''
    #* x を形式に丸め、保存用の code を int64 で返す: INT の値か、FP のビットの code。
    #? 重みを保存するか書き出すときに使う。
    '''
    
    grid = format_quantize(x, number_format, mode)
    if number_format.kind == "int":
        return integer_encode(grid, number_format)
    return fp_encode(grid, number_format)


def format_decode(codes: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* 保存用の code を float32 の値に戻す。
    '''
    
    if number_format.kind == "int":
        return integer_decode(codes, number_format)
    return fp_decode(codes, number_format)


def format_to_integer(values: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* 正確な整数の計算のために、グリッドの値を整数にする: value / step_min を int64 で返す。
    #? INT の値はそのまま。FP4 の値は 2 倍になる（刻みが 0.5）ので、1.5 は 3 になる。
    '''
    
    return torch.round(values / number_format.step_min).to(torch.int64)
