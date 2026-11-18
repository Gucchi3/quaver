# @file fp.py
# @brief 小さな FP の形式（FP4、UFP4、ブロックの scale の UE4M3）
# @author Kohei Yamaguchi, Claude

import torch

from .types import NumberFormat



def fp_format(exponent_bits: int, mantissa_bits: int, signed: bool, nan_top: bool, name: str = "") -> NumberFormat:
    '''
    #* 指数のビット数と仮数のビット数から、小さな FP の形式を作る。
    #? 符号なしの形式（UFP）は、符号のビットを仮数のビットに 1 つ足して使う: UFP4 は E2M2。name="" のときは、種類とビット数で名前を付ける（fp4、ufp4）。
    #! nan_top=True のときは一番上の code を NaN に取っておく（E4M3 と同じ）ので、一番大きな値は 1 刻み下になる。
    '''
    
    # 指数の範囲を求める
    bias     = 2 ** (exponent_bits - 1) - 1
    low_exp  = 1 - bias
    high_exp = 2**exponent_bits - 1 - bias
    
    # 一番大きな値、ビット数、名前を求める
    top_mantissa = 2**mantissa_bits - (2 if nan_top else 1)
    max_value    = 2.0**high_exp * (1 + top_mantissa / 2**mantissa_bits)
    bits         = exponent_bits + mantissa_bits + (1 if signed else 0)
    name         = name or f"{'fp' if signed else 'ufp'}{bits}"
    return NumberFormat(name=name, kind="fp", bits=bits, signed=signed, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits, exponent_bias=bias, nan_top=nan_top, min_value=-max_value if signed else 0.0, max_value=max_value,
                        step_min=2.0 ** (low_exp - mantissa_bits))


def fp_step(x: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* x の各値での、FP の形式のグリッドの刻みを返す。
    #? [2^e, 2^(e+1)) の値の刻みは 2^(e - mantissa_bits)。とても小さな値（subnormal）は、一番低い指数の刻みを使う。
    #! frexp と ldexp は指数しか変えないので、これは正確である。
    '''
    
    exponent = _fp_exponent(x, number_format)
    return torch.ldexp(torch.ones_like(x), exponent - number_format.mantissa_bits)


def fp_encode(values: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* FP のグリッドの値を、ビットの code（符号、指数、仮数）の int64 にする。
    #! 値は先にグリッドに乗せておく（たとえば format_quantize() で）。
    '''
    
    # 各絶対値の指数を求める
    size      = values.abs()
    exponent  = _fp_exponent(size, number_format)
    subnormal = size < 2.0 ** (1 - number_format.exponent_bias)
    
    # subnormal の指数の欄は 0 で、仮数は 2 つの 2 のべき乗の間の位置になる
    hidden   = torch.where(subnormal, 0, 2**number_format.mantissa_bits        )
    field    = torch.where(subnormal, 0, exponent + number_format.exponent_bias).to(torch.int64)
    mantissa = (torch.ldexp(size, number_format.mantissa_bits - exponent) - hidden).to(torch.int64)
    code     = (field << number_format.mantissa_bits) | mantissa
    
    # 符号付きの形式では、一番上に符号のビットを置く
    if not number_format.signed:
        return code
    return code | (torch.signbit(values).to(torch.int64) << (number_format.bits - 1))


def fp_decode(codes: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* FP のビットの code を float32 の値に戻す。
    #! nan_top の形式の一番上の code は NaN になる。
    '''
    
    # code から欄を取り出す
    mantissa_bits = number_format.mantissa_bits
    mantissa      = codes & (2**mantissa_bits - 1)
    field         = (codes >> mantissa_bits) & (2**number_format.exponent_bits - 1)
    
    # subnormal には隠れた 1 がなく、一番低い指数を使う
    hidden   = torch.where(field > 0, 2**mantissa_bits, 0)
    exponent = torch.clamp(field - number_format.exponent_bias, min=1 - number_format.exponent_bias)
    value    = torch.ldexp((hidden + mantissa).to(torch.float32), (exponent - mantissa_bits).to(torch.int32))
    
    # 符号を付け、NaN の code を NaN にする
    if number_format.signed:
        value = torch.where(((codes >> (number_format.bits - 1)) & 1) == 1, -value, value)
    if number_format.nan_top:
        top   = (field == 2**number_format.exponent_bits - 1) & (mantissa == 2**mantissa_bits - 1)
        value = torch.where(top, torch.nan, value)
    return value


def fp_values(number_format: NumberFormat) -> torch.Tensor:
    '''
    #* FP の形式が持てるすべての値を、小さい順に、NaN を除いて返す（-0 と 0 は 1 つの値として数える）。
    #? テストと表引きの表が使う。
    '''
    
    values = fp_decode(torch.arange(2**number_format.bits), number_format) + 0.0
    return torch.unique(values[~torch.isnan(values)])


def _fp_exponent(x: torch.Tensor, number_format: NumberFormat) -> torch.Tensor:
    '''
    #* 各値の floor(log2(|x|)) を返す。ただし、形式の一番低い指数より下にはしない。
    #? fp_step() と fp_encode() が使う。
    '''
    
    _, exponent = torch.frexp(x)
    return torch.clamp(exponent - 1, min=1 - number_format.exponent_bias)
