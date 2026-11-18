# @file params.py
# @brief 実数の値と整数の code の行き来と、training mode の計算の float の型
# @author Kohei Yamaguchi, Claude

import torch

from ..formats import RoundMode, build_format, device_define, format_quantize, format_to_integer, ste_attach
from .types    import BlockScale, QuantParams

MATH_DTYPES: dict[str, torch.dtype] = {
    "train": torch.float64,
    "test" : torch.float64,
}



def params_math_dtype() -> torch.dtype:
    '''
    #* 今の training mode の計算の float の型を返す: 勾配があるときは "train" の型、ないときは "test" の型。
    #? fake の op が呼ぶ。precision_set_math() が MATH_DTYPES を変える（初めはどちらも float64）。整数の op はいつも float64 を使う。
    #! torch.is_grad_enabled() は CPU のフラグを読むだけなので、GPU の同期はない。CUDA graph は、記録したときの型を使い続ける。
    '''
    
    return MATH_DTYPES["train" if torch.is_grad_enabled() else "test"]


def params_codes(x: torch.Tensor, params: QuantParams, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    '''
    #* fake quantize した値から、整数の code（int64）を取り戻す。
    #? fake の op が、integer mode と同じ整数の計算をするのに使う。型は params_math_dtype() のものを渡す。
    #! float64 では、x の float の誤差が code の刻みの半分より小さければ正確である（TF32 を止めておく）。float32 では、大きな累積で 1 ずれることがある。
    '''
    
    return torch.round(x.detach().to(dtype) / params.code_scale.to(dtype) + params.zero_point.to(dtype)).to(torch.int64)


def params_values(codes: torch.Tensor, params: QuantParams, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    '''
    #* 整数の code を、実数の float32 の値に戻す。
    #? dtype は計算の型。|code - zero point| が 2^24 より小さい間は、float32 でも float64 と同じ値になる。
    '''
    
    return ((codes.to(dtype) - params.zero_point.to(dtype)) * params.code_scale.to(dtype)).float()


def params_unit(params: QuantParams) -> torch.Tensor:
    '''
    #* 正確な和が使う整数の 1 刻みの実数の値を返す: code の scale。ブロックごとの scale のテンソルでは step_min x ブロックの scale の刻み x テンソルの scale。
    #? 整数は params_integers()。テンソルのブロックはこの 1 つの単位を共有するので、たくさんのブロックにわたる和も scale は 1 つになる。
    '''
    
    if params.block is None:
        return params.code_scale
    return params.block.tensor_scale * (params.number_format.step_min * params.block.scale_format.step_min)


def params_integers(codes: torch.Tensor, params: QuantParams) -> torch.Tensor:
    '''
    #* 正確な和が使う整数（int64）を返す: code - zero point。ブロックごとの scale のテンソルでは、さらにブロックの scale の code をかける。
    #? 実数の値 = 整数 x params_unit(params)。ブロックの scale が中に入っているので、違うブロックの値をそのまま足せる。
    '''
    
    centered = codes - params.zero_point.to(torch.int64)
    if params.block is None:
        return centered
    return centered * params.block.codes


def params_bits(params: QuantParams) -> int:
    '''
    #* params_integers() の |整数| の最大のビット数を、形式だけから（データを使わずに）返す。
    #? 正確な和が、動かす前にどれだけ広くなるかを知るのに使う。
    '''
    
    # ただのテンソル: code から zero point を引いた値は、形式の幅に収まる
    number_format = params.number_format
    largest       = round((number_format.max_value - number_format.min_value) / number_format.step_min)
    if params.block is None:
        return largest.bit_length()
    
    # ブロックごとの scale のテンソル: |code| の最大 x ブロックの scale の code の最大
    scale_format = params.block.scale_format
    top          = round(max(-number_format.min_value, number_format.max_value) / number_format.step_min)
    return (top * round(scale_format.max_value / scale_format.step_min)).bit_length()


def params_named(zero_point: torch.Tensor, number_format: str, block_codes: torch.Tensor | None = None, scale_format: str = "") -> QuantParams:
    '''
    #* zero point と形式の名前（とブロックの scale の code）から、scale が 1 の code のパラメータを作る。
    #? デバイスの op（torch.ops.quaver.*）はただのテンソルと名前を受け取り、整数の道具のために、これでパラメータを作り直す。
    #! scale は 1 なので、結果は scale が関係しない所だけに使う: 正確な和（params_integers、params_bits）と、code の単位での丸め（params_round）。
    '''
    
    one   = torch.ones((), device=zero_point.device)
    block = None if block_codes is None else BlockScale(block_codes, build_format(scale_format), one)
    return QuantParams(one, zero_point, build_format(number_format), block)


def params_note(codes: torch.Tensor, params: QuantParams) -> torch.Tensor:
    '''
    #* torch.export が動いている間は、code の形式、scale、zero point をグラフに入れ（op "note"）、code を返す。それ以外のときは、code をそのまま返す。
    #? op の integer の関数が、出力にこれを呼ぶ。model_export() は印を code の値に書いてノードを外すので、印はデバイスで何の負担にもならない。
    #! ブロックごとの scale のテンソルはテンソルの scale を、NEQ の出力は整数 p を印にする。ほかの scale はそのまま印にする。
    '''
    
    if not torch.compiler.is_exporting():
        return codes
    if params.block is not None:
        return torch.ops.quaver.note(codes, params.block.tensor_scale, params.zero_point, params.number_format.name, "tensor_scale")
    if params.peak is not None:
        return torch.ops.quaver.note(codes, params.peak, params.zero_point, params.number_format.name, "peak")
    return torch.ops.quaver.note(codes, params.scale, params.zero_point, params.number_format.name, "scale")


@device_define("note")
def params_note_device(codes: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, number_format: str, scale_name: str) -> torch.Tensor:
    '''
    #* op "note": code の写しで、その形式、scale（scale_name という名前）、zero point を引数に持つ。
    #? params_note() が、torch.export が動いている間だけ呼ぶ。書き出しがそれを読み、取り除く。
    '''
    
    return codes.clone()


def params_round(value: torch.Tensor, params: QuantParams, mode: RoundMode) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    #* code の単位の値（zero point を足す前）を形式の code に丸め、範囲の中にあった値を返す。
    #? integer mode のすべての op はここで終わるので、INT と FP の出力は同じ道を通る。mode は INT の形式用。FP の形式は、FP のハードウェア（IEEE、OCP MX）と同じく、いつも half to even で丸める。
    #! 正確な値を渡す。たとえば、整数の和や、整数と 2 のべき乗の積（float64 では 2^53 まで正確）。計算は value の float の型のまま行う。
    '''
    
    number_format = params.number_format
    grid          = value * number_format.step_min
    zero          = params.zero_point.to(value.dtype)
    inside        = (grid + zero >= number_format.min_value) & (grid + zero <= number_format.max_value)
    return format_to_integer(format_quantize(grid, number_format, mode if number_format.kind == "int" else "half_even", zero), number_format), inside


def params_attach(reference: torch.Tensor, codes: torch.Tensor, params: QuantParams, inside: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    '''
    #* forward では code の実数の値を返し、backward では reference を fake quantize したときの勾配を返す。
    #? fake の op が、整数の計算のあとで呼ぶ。reference は範囲の中で straight-through の勾配をもらう。学習する scale と zero point は LSQ の勾配をもらう。
    #! forward の値は params_values(codes, params, dtype) と完全に同じ。学習する scale は、LSQ の勾配の scale をかけたものを渡す（Quantizer.measure() がそうする）。
    '''
    
    # reference: 範囲の中で straight-through の勾配
    attached = ste_attach(reference, params_values(codes, params, dtype), inside)
    if not (params.scale.requires_grad or params.zero_point.requires_grad):
        return attached
    
    # 学習する scale と zero point: 範囲の中では d/dscale = code - reference / scale、範囲の外では範囲の端（LSQ）。code は整数の計算のもの
    number_format = params.number_format
    scaled        = reference.detach() / params.scale + params.zero_point
    clipped       = torch.where(inside, scaled, scaled.clamp(number_format.min_value, number_format.max_value).detach())
    grid          = codes.to(scaled.dtype) * number_format.step_min
    learned       = (clipped + (grid - clipped).detach() - params.zero_point) * params.scale
    return attached + (learned - learned.detach())
