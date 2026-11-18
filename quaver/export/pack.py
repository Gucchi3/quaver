# @file pack.py
# @brief code をデバイスのメモリと同じ形でバイトに詰める: 2 bit と 4 bit の code は下のビットから詰め、それより広い code はリトルエンディアンにし、FP の code はビットのパターンで置く
# @author Kohei Yamaguchi, Claude

from typing import Any

import torch

from ..formats import build_format, fp_decode, fp_encode

# メモリでの幅: 2 bit と 4 bit はバイトに詰め、ほかの幅は 8、16、32、64 bit のうち次に大きいものを使う（int6 は 8 bit、uint10 は 16 bit）
PACK_WIDTHS = (2, 4, 8, 16, 32, 64)



def pack_codes(codes: torch.Tensor, number_format: str) -> tuple[bytes, dict[str, Any]]:
    '''
    #* code（要素の順）をバイトに詰め、置き方（encoding と、ビットでの幅）を返す。
    #? INT の code は、その幅の 2 の補数で置く（"int"）。FP の code（quaver ではグリッドの整数）は、ビットのパターンで置く: fp4 は E2M1、ufp4 は E2M2、ue4m3 は E4M3（"fp"）。
    #? "float32" と "float64" は float を IEEE 754 で置く（"ieee754"）。2 bit と 4 bit の値は、各バイトを下のビットから埋める: 前の要素が下のビットに入る。
    #! 形式の範囲の外の値は ValueError になる。numpy が要る（"export" の extra）。
    '''
    
    # numpy はバイトを決まった順（リトルエンディアン）で書くので、ファイルはコンピュータによらない
    import numpy
    
    # float は IEEE 754 で置く
    flat = codes.detach().cpu().reshape(-1)
    if number_format in ("float32", "float64"):
        bits = 32 if number_format == "float32" else 64
        return flat.double().numpy().astype(f"<f{bits // 8}").tobytes(), {"encoding": "ieee754", "bits": bits}
    
    # 置く整数: INT の code はそのまま、FP の code はビットのパターン（符号、指数、仮数）
    named = build_format(number_format)
    if named.kind == "fp":
        stored = fp_encode(flat.double() * named.step_min, named)
        info   = {"encoding": "fp", "signed": named.signed, "exponent_bits": named.exponent_bits, "mantissa_bits": named.mantissa_bits, "exponent_bias": named.exponent_bias}
        low    = 0
        high   = 2**named.bits - 1
    else:
        stored = flat.long()
        info   = {"encoding": "int", "signed": named.signed}
        low    = round(named.min_value)
        high   = round(named.max_value)
    if stored.numel() and (int(stored.min()) < low or int(stored.max()) > high):
        raise ValueError(f"Codes out of the range of {number_format}: {int(stored.min())} to {int(stored.max())}.")
    if named.kind == "fp" and not torch.equal(fp_decode(stored, named).double() / named.step_min, flat.double()):
        raise ValueError(f"Codes that are not on the grid of {number_format}.")
    
    # 広い値: リトルエンディアンで、符号付きの INT は 2 の補数
    width = next(width for width in PACK_WIDTHS if width >= named.bits and (width >= 8 or width == named.bits))
    if width >= 8:
        kind = "i" if info["signed"] and named.kind == "int" else "u"
        return stored.numpy().astype(f"<{kind}{width // 8}").tobytes(), {**info, "bits": width}
    
    # 狭い値: 各値の下のビット（2 の補数）を、1 バイトに複数個、下のビットから詰める
    count  = 8 // width
    values = (stored & (2**width - 1)).numpy().astype("uint8")
    values = numpy.pad(values, (0, -len(values) % count)).reshape(-1, count)
    packed = numpy.zeros(len(values), dtype="uint8")
    for index in range(count):
        packed |= values[:, index] << (index * width)
    return packed.tobytes(), {**info, "bits": width}
