# @file integer.py
# @brief INT の形式（int2〜int8、uint2〜uint8、int32）
# @author Kohei Yamaguchi, Claude

import torch

from .types import NumberFormat



def integer_format(bits: int, signed: bool) -> NumberFormat:
    '''
    #* INT の形式を作る: 符号付きなら int2〜int8、符号なしなら uint2〜uint8。
    #? バイアスと累積は、中で int32 を使う。
    '''
    
    low  = -(2 ** (bits - 1)) if signed else 0
    high = 2 ** (bits - 1) - 1 if signed else 2**bits - 1
    name = f"{'int' if signed else 'uint'}{bits}"
    return NumberFormat(name=name, kind="int", bits=bits, signed=signed, exponent_bits=0, mantissa_bits=0, exponent_bias=0, nan_top=False, min_value=float(low), max_value=float(high), step_min=1.0)


def integer_encode(values: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* INT のグリッドの値を int64 の code にする。
    #! 値は先にグリッドに乗せておく（たとえば format_quantize() で）。
    '''
    
    return values.to(torch.int64)


def integer_decode(codes: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* INT の code を float32 の値に戻す。
    '''
    
    return codes.to(torch.float32)
