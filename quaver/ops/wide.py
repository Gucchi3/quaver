# @file wide.py
# @brief 広い積（約 80 bit まで）を int64 だけで正確に丸める。wide の累積とブロックの scale のため
# @author Kohei Yamaguchi, Claude

import torch

from ..formats import NumberFormat, RoundMode

# a x multiplier は high x 2^WIDE_SPLIT + low（0 <= low < 2^WIDE_SPLIT）の形で持つ
WIDE_SPLIT = 32
WIDE_LIMIT = 2**61



def wide_round(a: torch.Tensor, multiplier: torch.Tensor, divisor: torch.Tensor, shift: torch.Tensor, mode: RoundMode) -> torch.Tensor:
    '''
    #* round(a x multiplier / (divisor x 2^shift)) を、int64 の計算だけで正確に返す（テンソルは broadcast する）。
    #? 積 a x multiplier は 82 bit まで要ることがあるので、2 つの部分に分けて持つ。float64 では正確にならない wide の累積とブロックの scale が使う。
    #! |a| < 2^62、0 <= multiplier < 2^20、1 <= divisor < 2^20、-30 <= shift（94 を超える shift は 94 として数える）。2^40 以上の結果は正確でなく、大きいままになるだけなので、あとで clamp する。
    '''
    
    # 入力を、同じ形の int64 にする（94 を超える shift は 94 として数える。そのとき結果は 0）
    a, multiplier, divisor, shift = torch.broadcast_tensors(a.long(), multiplier.long(), divisor.long(), shift.long().clamp(max=94))
    
    # a x multiplier = high x 2^32 + low、0 <= low < 2^32（a = a_high x 2^32 + a_low とし、a_low x multiplier の繰り上がりを high に入れる）
    mask = 2**WIDE_SPLIT - 1
    low  = (a & mask) * multiplier
    high = (a >> WIDE_SPLIT) * multiplier + (low >> WIDE_SPLIT)
    low  = low & mask
    
    # 32 以上の shift: 整数の部分は high から求め、残りは (rest_high, low) で、rest_high < 2^(shift - 32)
    above      = (shift - WIDE_SPLIT).clamp(min=0)
    whole_wide = high >> above
    rest_high  = high - (whole_wide << above)
    
    # 32 より小さな shift で、|a x multiplier| < 2^61 のとき: 積を 1 つの int64 として求め、shift する（WIDE_LIMIT を超える左シフトは大きいままになるだけ）
    ones        = torch.ones_like(shift)
    small       = high.abs() < 2**29
    product     = (torch.where(small, high, 0) << WIDE_SPLIT) + low
    grows       = product.abs() < (WIDE_LIMIT >> (-shift).clamp(min=0, max=62))
    left        = torch.where(grows, product, 0) << (-shift).clamp(min=0)
    low_bits    = (ones << shift.clamp(min=1, max=WIDE_SPLIT - 1)) - 1
    whole_small = torch.where(shift >= 0, product >> shift.clamp(min=0, max=WIDE_SPLIT - 1), torch.where(grows, left, torch.where(product < 0, -WIDE_LIMIT, WIDE_LIMIT)))
    rest_small  = torch.where(shift > 0 , product & low_bits                               , 0                                                                          )
    
    # 32 より小さな shift で、|a x multiplier| >= 2^61 のとき: high を上へずらし（WIDE_LIMIT を超えると結果は大きいままになるだけ）、low が残りを与える
    below      = (WIDE_SPLIT - shift).clamp(min=0, max=62)
    safe       = high.abs() < (WIDE_LIMIT >> below)
    moved      = (torch.where(safe, high, 0) << below) + (low >> shift.clamp(min=0, max=WIDE_SPLIT - 1))
    whole_big  = torch.where(safe     , moved         , torch.where(high < 0, -WIDE_LIMIT, WIDE_LIMIT))
    rest_big   = torch.where(shift > 0, low & low_bits, 0                                             )
    whole_near = torch.where(small    , whole_small   , whole_big                                     )
    rest_near  = torch.where(small    , rest_small    , rest_big                                      )
    
    # 2^shift で割った整数の部分を求め、次に divisor で割る: value = quotient + (remainder x 2^shift + rest) / (divisor x 2^shift)
    wide      = shift >= WIDE_SPLIT
    whole     = torch.where(wide, whole_wide, whole_near)
    quotient  = torch.div(whole, divisor, rounding_mode="floor")
    remainder = whole - quotient * divisor
    
    # 残りを 2^shift の半分と比べる: 小さければ -1、等しければ 0、大きければ 1（(rest_high, low) の残りは rest_high x 2^32 + low）
    half_high   = torch.where(above > 0             , ones << (above - 1).clamp(min=0)                                              , 0                         )
    half_low    = torch.where(above > 0             , 0                                                                             , 2 ** (WIDE_SPLIT - 1)     )
    compare_far = torch.where(rest_high != half_high, torch.sign(rest_high - half_high)                                             , torch.sign(low - half_low))
    compare_mid = torch.where(shift > 0             , torch.sign(rest_near - (ones << (shift - 1).clamp(min=0, max=WIDE_SPLIT - 1))), -1                        )
    compare     = torch.where(wide                  , compare_far                                                                   , compare_mid               )
    rest        = torch.where(wide                  , (rest_high != 0) | (low != 0)                                                 , rest_near != 0            )
    
    # 小数の部分の 2 倍を 1 と比べる: (2 x remainder - divisor) x 2^shift + 2 x rest。大きい、ちょうど半分、小さいのどれか
    double  = 2 * remainder - divisor
    greater = (double >= 1) | ((double == -1) & (compare > 0)) | ((double == 0) & rest)
    tie     = ((double == -1) & (compare == 0)) | ((double == 0) & ~rest)
    
    # mode で丸める（floor は決して上へ行かない）
    if mode == "floor":
        return quotient
    tie_up = {"half_up": torch.ones_like(tie), "half_even": quotient % 2 == 1, "half_away": a >= 0}[mode]
    return quotient + (greater | (tie & tie_up)).long()


def wide_round_format(a: torch.Tensor, multiplier: torch.Tensor, divisor: torch.Tensor, shift: torch.Tensor, number_format: NumberFormat, mode: RoundMode, zero_point: torch.Tensor | None = None) -> torch.Tensor:
    '''
    #* code の単位の値 a x multiplier / (divisor x 2^shift) を、number_format の code に正確に丸め、zero point を足して、範囲の両端で止める。
    #? INT の形式は mode で丸める。FP の形式は、format_quantize() と同じく、グリッド（2 のべき乗ごとに刻みが 2 倍になる）の上で、大きさを half to even で丸めてから符号を付ける。
    #! 制限は wide_round() と同じ。FP のグリッドには大きさが要るので、先に floor で 2 のべき乗を求める。
    '''
    
    # INT の形式: 丸め、zero point を足し、clamp する
    low  = round(number_format.min_value / number_format.step_min)
    high = round(number_format.max_value / number_format.step_min)
    zero = torch.zeros_like(a) if zero_point is None else zero_point.long()
    if number_format.kind == "int":
        return (wide_round(a, multiplier, divisor, shift, mode).clamp(-WIDE_LIMIT, WIDE_LIMIT) + zero).clamp(low, high)
    
    # FP の形式: 大きさの 2 のべき乗（上限以上の code は clamp するだけ）と、そこでの刻み（2^(mantissa + 1) までは 1、そのあとは 2 倍ずつ）
    size  = a.abs()
    whole = wide_round(size, multiplier, divisor, shift, "floor").clamp(0, high)
    power = torch.frexp(whole.double())[1].long() - 1
    steps = (power - number_format.mantissa_bits).clamp(min=0)
    
    # その刻みで half to even に丸め、符号を戻して clamp する
    codes = wide_round(size, multiplier, divisor, shift + steps, "half_even").clamp(0, high) << steps
    return torch.where(a < 0, -codes, codes).clamp(low, high)
