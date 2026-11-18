# @file block.py
# @brief デバイスと同じく、ブロックごとの scale の形式（NVFP4 など）に量子化する: ブロックの最大からブロックの scale を、次に要素を、整数だけで求める
# @author Kohei Yamaguchi, Claude

import torch

from ..formats   import NumberFormat, build_format, device_define, format_to_integer
from ..scales    import BlockScale, Granularity, GranularityKind, QuantParams, compute_block, group_merge, group_split, params_attach, params_codes, params_math_dtype, params_note
from .requantize import requantize_params
from .wide       import wide_round_format



def block_quantize_fake(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, scale_format: NumberFormat, tensor_scale: torch.Tensor, granularity: Granularity, style_name: str) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* training mode: block_quantize_integer の整数の計算で、ブロックごとの scale の形式に量子化する。勾配はそのまま通す。
    #? x は、テンソルで scale が 1 つの code の正確な値（ほかの op の出力）でなければならないので、TF32 を止めておく。
    #! 勾配は、値がそのブロックの上限（形式の上限 x ブロックの scale）の下に収まる所を通る。
    '''
    
    dtype         = params_math_dtype()
    codes, params = block_quantize_integer(params_codes(x, x_params, dtype), x_params, number_format, scale_format, tensor_scale, granularity, style_name)
    inside        = x.abs() <= number_format.max_value * params.scale
    return params_attach(x, codes, params, inside, dtype), params


def block_quantize_integer(q: torch.Tensor, q_params: QuantParams, number_format: NumberFormat, scale_format: NumberFormat, tensor_scale: torch.Tensor, granularity: Granularity, style_name: str) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* integer mode: 静的な scale が 1 つの code（形式の code か、どの幅の累積でもよい）を、デバイスが動きながらするのと同じく、ブロックごとの scale の形式に量子化する。
    #? ブロックの scale の code = |ブロックで一番大きな code| x M6 / 2^n6 を、scale_format に half to even で丸めたもの。要素の code = code x Mr / (ブロックの scale の code x 2^nr) を half to even で丸めたもの。
    #? デバイスでは、ブロックの code での割り算を、グリッドの中点 x ブロックの code との比較にできる。デバイスの op torch.ops.quaver.block_quantize が計算し、M6、n6、Mr、nr はここで求める。
    #! q_params の scale はテンソルで 1 つでなければならない。整数だけを使い、q がどの幅でも正確（wide_round_format）。
    '''
    
    # q の 1 刻みと、ブロックの scale の 1 刻みの静的な比、要素の 1 刻みとの静的な比（ブロックの scale の刻みで数える）
    step     = tensor_scale.double() * scale_format.step_min
    to_scale = requantize_params(q_params.code_scale.double() / (number_format.max_value * step), style_name)
    to_value = requantize_params(q_params.code_scale.double() / (number_format.step_min * step) , style_name)
    
    # 要素とそのブロックの scale の code と、各値の scale
    codes, scales = torch.ops.quaver.block_quantize(q, q_params.zero_point.to(torch.int64), to_scale.multiplier, to_scale.shift, to_value.multiplier, to_value.shift, number_format.name, scale_format.name, granularity.kind, granularity.axis,
                                                  granularity.block_size)
    scale         = scales.to(torch.float32) * scale_format.step_min * tensor_scale
    params        = QuantParams(scale, torch.zeros([1] * scale.ndim, device=scale.device), number_format, BlockScale(scales, scale_format, tensor_scale))
    return params_note(codes, params), params


@device_define("block_quantize")
def block_quantize_device(q: torch.Tensor, zero_point: torch.Tensor, scale_multiplier: torch.Tensor, scale_shift: torch.Tensor, value_multiplier: torch.Tensor, value_shift: torch.Tensor, out_format: str, scale_format: str, kind: GranularityKind,
                          axis: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* デバイスの op "block_quantize": 各ブロックの scale の code をそのブロックの |code| の最大から求め、次に各要素をそのブロックの code で割る。どちらも half to even で丸める。
    #? block_quantize_integer() が torch.ops.quaver.block_quantize として呼ぶ。要素の code と、q に broadcast できるように広げ直したブロックの code を返す。
    #! multiplier と shift は静的: scale_* は q の 1 刻みを out_format の上限でのブロックの scale の刻みに、value_* は要素の刻みに直す。
    '''
    
    # 各ブロックの scale の code を、ブロックの |code| の最大から求める（一番小さな刻みより下にはしない）
    granularity = Granularity(kind, axis, block_size)
    centered    = q - zero_point
    largest     = group_split(centered.abs(), granularity).amax(dim=1)
    scales      = wide_round_format(largest, scale_multiplier, torch.ones_like(largest), scale_shift, build_format(scale_format), "half_even").clamp(min=1)
    scales      = group_merge(scales, q.shape, granularity)
    
    # 各要素をそのブロックの scale の code で割り、形式に half to even で丸める
    return wide_round_format(centered, value_multiplier, scales, value_shift, build_format(out_format), "half_even"), scales


def block_quantize_float(x: torch.Tensor, x_params: QuantParams, number_format: NumberFormat, scale_format: NumberFormat, tensor_scale: torch.Tensor, granularity: Granularity, style_name: str) -> tuple[torch.Tensor, QuantParams]:
    '''
    #* float のモード（calibration）: x をそのまま返し、float の値がもらうはずのブロックの scale も返す。
    '''
    
    block = compute_block(group_split(x.detach(), granularity).abs().amax(dim=1), tensor_scale, number_format, scale_format)
    codes = group_merge(format_to_integer(block, scale_format), x.shape, granularity)
    scale = group_merge(block * tensor_scale                  , x.shape, granularity)
    return x, QuantParams(scale, torch.zeros_like(scale), number_format, BlockScale(codes, scale_format, tensor_scale))
