# @file compute.py
# @brief 値の範囲を scale と zero point に変える
# @author Kohei Yamaguchi, Claude

import torch

from ..formats import NumberFormat, format_quantize, round_half_even

SCALE_MIN = torch.finfo(torch.float32).tiny
MSE_STEPS = 100



def compute_scale(low: torch.Tensor, high: torch.Tensor, number_format: NumberFormat, symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* 値の範囲を scale と zero point に変える。
    #? 対称はどの形式でも使える。非対称は INT の形式用。
    #! 非対称の範囲は 0 を含むように広げるので、0 は正確なままになる。
    '''
    
    # 対称: |low| と |high| の大きい方を形式の一番大きな値に合わせ、zero point は 0 にする
    if symmetric:
        scale = (torch.maximum(low.abs(), high.abs()) / number_format.max_value).clamp_min(SCALE_MIN)
        return scale, torch.zeros_like(scale)
    
    # 非対称: 範囲が 0 を含むようにし、すべての code に広げて、実数の 0 を整数の code に乗せる
    low        = low.clamp(max=0.0)
    high       = high.clamp(min=0.0)
    scale      = ((high - low) / (number_format.max_value - number_format.min_value)).clamp_min(SCALE_MIN)
    zero_point = round_half_even(number_format.min_value - low / scale).clamp(number_format.min_value, number_format.max_value)
    return scale, zero_point


def compute_mse_range(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor, number_format: NumberFormat, symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* 各グループの範囲を、量子化した値の二乗誤差が一番小さくなる範囲に縮める（LSQ+ の MSE の初期値）。
    #? values はグループごとに 1 行を持ち、low と high は測った範囲。対称の範囲は両端を一緒に縮め、非対称の範囲は下の端と上の端を交互に縮める（2 回ずつ）。
    #! 測った端に MSE_STEPS 通りの割合をかけて試し、half to even で丸める。GPU の同期をしないので、最初の学習の step の中で動かせる。
    '''
    
    # 試す割合、各回で動かす端、実際に使う範囲（非対称の範囲は 0 を含む）
    fractions = torch.arange(1, MSE_STEPS + 1, dtype=values.dtype, device=values.device) / MSE_STEPS
    turns     = ["both"] if symmetric else ["low", "high", "low", "high"]
    low, high = (low, high) if symmetric else (low.clamp(max=0.0), high.clamp(min=0.0))
    best      = (low, high)
    for turn in turns:
        # 動かす端の各割合を試し、グループごとに誤差が一番小さい範囲を残す
        start = best
        error = torch.full_like(low, torch.inf)
        for fraction in fractions:
            trial       = (start[0] if turn == "high" else low * fraction, start[1] if turn == "low" else high * fraction)
            scale, zero = compute_scale(trial[0], trial[1], number_format, symmetric)
            quantized   = (format_quantize(values / scale[:, None], number_format, "half_even", zero[:, None]) - zero[:, None]) * scale[:, None]
            trial_error = (quantized - values).square().mean(dim=1)
            better      = trial_error < error
            error       = torch.where(better, trial_error, error)
            best        = (torch.where(better, trial[0], best[0]), torch.where(better, trial[1], best[1]))
    return best


def compute_pow2(scale: torch.Tensor) -> torch.Tensor:
    '''
    #* shift でしか割れないデバイスのために、scale を 2 のべき乗に切り上げる。
    #! frexp と ldexp は指数しか変えないので、これは正確である。
    '''
    
    mantissa, exponent = torch.frexp(scale)
    return torch.ldexp(torch.ones_like(scale), exponent - (mantissa == 0.5).to(exponent.dtype))


def compute_block(block_amax: torch.Tensor, tensor_scale: torch.Tensor, number_format: NumberFormat, scale_format: NumberFormat) -> torch.Tensor:
    '''
    #* ブロックごとの scale の形式の、各ブロックの scale を返す: block_amax / top / テンソルの scale を scale_format に丸めたもの（half to even）。
    #? top は number_format の一番大きな値（fp4 は 6、ufp4 は 7）。scale_format が ue4m3 なら NVFP4、符号なしの INT の形式（uint6、uint10）なら整数のブロックの scale（VS-Quant）になる。
    #! ブロックの scale は scale_format の一番小さな刻み以上に保つので、0 だけのブロックも scale を持つ。
    '''
    
    return format_quantize(block_amax / number_format.max_value / tensor_scale, scale_format, "half_even").clamp_min(scale_format.step_min)
