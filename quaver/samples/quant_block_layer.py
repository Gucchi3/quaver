# @file quant_block_layer.py
# @brief 整数の計算だけで動く、ブロックごとの scale の conv と linear のサンプルの層（NVFP4 など）に共通の forward
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats    import build_format, ste_attach
from ..ops        import ActivationKind, accumulate_format, accumulate_scale, activation_float, activation_integer, bias_quantize, block_quantize_float, block_quantize_integer, mode_ops, requantize_integer
from ..quantizers import Quantizer, Scaler
from ..scales     import BlockScaler, Granularity, QuantParams, RangeScaler, params_codes, params_values



class QuantBlockLayer(nn.Module):
    '''
    #* ブロックごとの scale の conv と linear のサンプルの層に共通の forward: 入力のチャネルのブロックごとの scale（重みと入力）、正確な wide の累積、出力。
    #? QuantBlockConv2d と QuantBlockLinear が、float の op（_layer_float）と正確な整数の op（_layer_integer）を書く。scale_format "ue4m3" で NVFP4 に、"uint6" か "uint10" で整数の scale になる。
    #! training mode は integer mode の整数の計算を code の上で動かし（累積は float の値には広すぎる）、勾配は float の層から求める。
    '''
    
    output_axis: int = 1
    activation : ActivationKind
    
    def __init__(self, layer: nn.Conv2d | nn.Linear, weight_format: str, scale_format: str, input_format: str | None, output_format: str, output_block: bool, activation: ActivationKind, style: str, block_size: int,
                 output_scaler: Scaler | None) -> None:
        '''
        #* float の層の重みとバイアスを受け取り、重み、入力のブロック、出力の量子化器を作る。
        #? input_format があるときは、ここで入力をブロックに量子化する（静的なテンソルの scale と、ブロックごとの scale）。input_format=None のときは、入力をそのまま使う（ブロックごとの scale か、W4A8 なら INT8）。
        #? output_block=True のときは output_format のブロックごとの scale の出力を出し（次のブロックごとの scale の層用）、False のときは静的な scale 1 つで output_format に requantize する。
        #? output_scaler は、その 1 つの scale をほかのテンソルと共有する（ブロックの残差のストリームで SharedScaler を 1 つ: scale の共有）。None のときは、この出力に従う RangeScaler を作る。
        #! 要素は half to even で丸める。入力のチャネルは block_size の倍数でなければならない。
        '''
        
        # この入口で確かめる: ブロックごとの scale の出力のブロックの scale はブロックごとに決まるので、その出力は 1 つの scale を共有できない
        if output_block and output_scaler is not None:
            raise ValueError("output_scaler shares one scale of an INT output, but output_block=True gives block scales that follow each block. Use output_block=False.")
        
        # 重みとバイアスは float の層と同じ名前にするので、その state_dict をそのまま読み込める
        super().__init__()
        self.weight = layer.weight
        self.bias   = layer.bias
        
        # 入力のチャネルのブロックごとに scale を 1 つ: 重みの軸 1 と、活性化のチャネルの軸
        self.weight_blocks    = Granularity("block", axis=1               , block_size=block_size)
        self.blocks           = Granularity("block", axis=self.output_axis, block_size=block_size)
        self.weight_quantizer = Quantizer(build_format(weight_format), BlockScaler(scale_format), self.weight_blocks, "half_even")
        self.input_format     = None if input_format is None else build_format(input_format)
        self.input_scaler     = BlockScaler(scale_format, RangeScaler())
        
        # 出力: 静的なテンソルの scale を持つブロックごとの scale（output_block_scaler）か、最大に従うか共有する静的な scale 1 つ（output_quantizer）。どちらか 1 つだけを使う
        self.output_format       = build_format(output_format)
        self.output_block        = output_block
        self.output_block_scaler = BlockScaler(scale_format, RangeScaler())
        self.output_quantizer    = Quantizer(self.output_format, RangeScaler() if output_scaler is None else output_scaler, Granularity(), batch_axis=0)
        
        # 層の設定
        self.activation   = activation
        self.style        = style
        self.integer_mode = False
        self.float_mode   = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* 層を動かす: integer mode では整数の code、training mode では fake の値（同じ整数の計算）、float のモードでは float の値を使う。
        #? input_format があるときは、ここで入力をブロックに量子化する。出力は次の op の形式で、wide の累積をそのまま出すことはない。
        '''
        
        # 入力: 静的なテンソルの scale を持つブロック（学習中は scaler が入力に従う）か、入力そのまま
        ops = mode_ops(self.integer_mode, self.float_mode)
        if self.input_format is not None:
            tensor_scale = self.input_scaler.tensor_scale(x, self.input_format, not self.integer_mode)
            x, x_params  = ops.block_quantize(x, x_params, self.input_format, self.input_scaler.scale_format, tensor_scale, self.blocks, self.style)
        
        # 重み: integer mode では code、training mode では fake の値、float のモードでは float の重み
        weight, w_params = (self.weight_quantizer.encode if self.integer_mode else self.weight_quantizer)(self.weight)
        acc_params       = QuantParams(accumulate_scale(x_params, w_params), torch.zeros(1, device=weight.device), accumulate_format(x_params, w_params))
        bias, bias_codes = (None, None) if self.bias is None else bias_quantize(self.bias, x_params, w_params, "half_away")
        view             = [-1 if axis == self.output_axis % x.ndim else 1 for axis in range(x.ndim)]
        
        # float のモード: float の層を動かし、出力の scaler にその値を見せる
        if self.float_mode:
            acc = activation_float(self._layer_float(x, weight, self.bias), acc_params, self.activation)
            if self.output_block:
                tensor_scale = self.output_block_scaler.tensor_scale(acc, self.output_format, True)
                return block_quantize_float(acc, acc_params, self.output_format, self.output_block_scaler.scale_format, tensor_scale, self.blocks, self.style)
            return acc, self.output_quantizer.measure(acc)
        
        # code から求めた正確な累積（training mode は fake の値から code を取り戻す）、バイアス、累積への活性化
        q_x = x if self.integer_mode else params_codes(x, x_params)
        q_w = weight if self.integer_mode else params_codes(weight, w_params)
        acc = self._layer_integer(q_x, x_params, q_w, w_params)
        acc = acc if bias_codes is None else acc + bias_codes.reshape(view)
        acc = activation_integer(acc, acc_params, self.activation)
        
        # 出力の code: 静的なテンソルの scale を持つブロック（学習中は scaler が実数の値に従う）か、静的な scale 1 つへの requantize
        real = acc.double() * acc_params.scale.double()
        if self.output_block:
            tensor_scale      = self.output_block_scaler.tensor_scale(real, self.output_format, not self.integer_mode)
            codes, out_params = block_quantize_integer(acc, acc_params, self.output_format, self.output_block_scaler.scale_format, tensor_scale, self.blocks, self.style)
            inside            = real.abs() <= self.output_format.max_value * out_params.scale
        else:
            out_params = self.output_quantizer.params(real) if self.integer_mode else self.output_quantizer.measure(real)
            codes      = requantize_integer(acc, acc_params, out_params, self.style)
            grid       = real / out_params.code_scale + out_params.zero_point
            inside     = (grid >= self.output_format.min_value / self.output_format.step_min) & (grid <= self.output_format.max_value / self.output_format.step_min)
        if self.integer_mode:
            return codes, out_params
        
        # training mode: forward では整数の計算の値を、backward では float の層の勾配（fake の入力、重み、バイアスで求めたもの）を使う
        reference = activation_float(self._layer_float(x, weight, bias), acc_params, self.activation)
        return ste_attach(reference, params_values(codes, out_params), inside), out_params
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* training mode と float のモード: fake quantize した（または float の）値で、層の float の op をする。
        #? QuantBlockConv2d と QuantBlockLinear がこれを書く。
        '''
        
        raise NotImplementedError
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* integer mode: 整数の code で、層の正確な op をする。
        #? QuantBlockConv2d と QuantBlockLinear がこれを書く。
        '''
        
        raise NotImplementedError
