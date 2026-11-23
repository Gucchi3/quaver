# @file quant_layer.py
# @brief conv と linear のサンプルの層に共通の forward
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats    import RoundMode, build_format, format_round_mode
from ..ops        import ACCUMULATOR_FORMAT, ActivationKind, accumulate_scale, bias_quantize, mode_ops
from ..quantizers import Quantizer, Scaler
from ..scales     import DynamicScaler, Granularity, QuantParams, RangeScaler



class QuantLayer(nn.Module):
    '''
    #* conv と linear のサンプルの層に共通の forward: 重み、INT32 のバイアス、累積、活性化、requantize。
    #? QuantConv2d と QuantLinear が、float の op（_layer_float）と正確な整数の op（_layer_integer）を書く。
    #! デバイスと同じく整数の code を受け取って返すには、integer_mode = True にする（または switch_integer_mode() を呼ぶ）。switch_quantizers(model, False) は float_mode = True にする（calibration）。
    '''
    
    output_axis: int = 1
    activation : ActivationKind
    round_mode : RoundMode | None
    
    def __init__(self, layer: nn.Conv2d | nn.Linear, weight_format: str, output_format: str | None, activation: ActivationKind, style: str, round_mode: RoundMode | None, weight_symmetric: bool, output_symmetric: bool,
                 output_scaler: Scaler | None) -> None:
        '''
        #* float の層の重みとバイアスを受け取り、重みと出力の量子化器を作る。
        #? float の層が PyTorch のいつもの初期値を与える。output_format=None のときは、requantize せずに INT32 の累積を返す。round_mode=None のときは、各形式のいつもの丸めを使う。
        #? output_scaler は、出力にほかのテンソルと共有する scale を与える（ブロックの残差のストリームで SharedScaler を 1 つ: scale の共有）。None のときは、この出力に従う RangeScaler を作る。
        #! weight_symmetric と output_symmetric で、対称の scale（zero point 0）か非対称の scale（zero point あり。INT の形式だけ）を選ぶ。
        '''
        
        # 重みとバイアスは float の層と同じ名前にするので、その state_dict をそのまま読み込める
        super().__init__()
        self.weight = layer.weight
        self.bias   = layer.bias
        
        # 毎回測る、出力チャネルごとの重みの scale と、範囲に従うか共有する出力の scale 1 つ（出力の軸 0 は batch）
        self.weight_quantizer = Quantizer(build_format(weight_format), DynamicScaler(symmetric=weight_symmetric), Granularity("channel", axis=0), round_mode)
        self.output_quantizer = None if output_format is None else Quantizer(build_format(output_format), RangeScaler(symmetric=output_symmetric) if output_scaler is None else output_scaler, Granularity(), round_mode, batch_axis=0)
        
        # 層の設定
        self.activation   = activation
        self.style        = style
        self.round_mode   = round_mode
        self.integer_mode = False
        self.float_mode   = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* 層を動かす: training mode では fake quantize した値、integer mode では整数の code、float のモードでは float の値を使う。
        #? 累積のあと、training mode と integer mode は同じ整数の計算をするので、出力は同じになる。float のモードは float の重みとバイアスを使う。
        '''
        
        # 重みを量子化する: training mode では fake の値、integer mode では code、量子化器が切ってあるときは float の重み
        ops              = mode_ops(self.integer_mode, self.float_mode)
        quantize         = self.weight_quantizer.encode if self.integer_mode else self.weight_quantizer
        weight, w_params = quantize(self.weight)
        
        # float の op か正確な整数の op で累積し、INT32 のバイアスを足す（float のモードは float のバイアスを足す）
        if self.bias is None:
            acc = self._layer_integer(x, x_params, weight, w_params) if self.integer_mode else self._layer_float(x, weight, None)
        else:
            bias, bias_codes = bias_quantize(self.bias, x_params, w_params, format_round_mode(ACCUMULATOR_FORMAT, self.round_mode))
            bias             = self.bias if self.float_mode else bias
            acc              = self._layer_integer(x, x_params, weight, w_params) + self._layer_view(bias_codes, x.ndim) if self.integer_mode else self._layer_float(x, weight, bias)
        
        # 累積は出力チャネルごとに scale を 1 つ持つ。それに活性化をかける
        scale      = self._layer_view(accumulate_scale(x_params, w_params), acc.ndim)
        acc_params = QuantParams(scale, torch.zeros_like(scale), ACCUMULATOR_FORMAT)
        acc        = ops.activation(acc, acc_params, self.activation)
        if self.output_quantizer is None:
            return acc, acc_params
        
        # デバイスと同じく、出力の形式に requantize する
        out_params = self.output_quantizer.params(acc) if self.integer_mode else self.output_quantizer.measure(acc)
        return ops.requantize(acc, acc_params, out_params, self.style), out_params
    
    def _layer_float(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        '''
        #* training mode と float のモード: fake quantize した（または float の）値で、層の float の op をする。
        #? QuantConv2d と QuantLinear がこれを書く。
        '''
        
        raise NotImplementedError
    
    def _layer_integer(self, q_x: torch.Tensor, x_params: QuantParams, q_w: torch.Tensor, w_params: QuantParams) -> torch.Tensor:
        '''
        #* integer mode: 整数の code で、層の正確な op をする。
        #? QuantConv2d と QuantLinear がこれを書く。
        '''
        
        raise NotImplementedError
    
    def _layer_view(self, values: torch.Tensor, ndim: int) -> torch.Tensor:
        '''
        #* 出力チャネルごとに 1 つの値を、出力に broadcast できる形にする。
        #? forward() が、バイアスと scale に使う。
        '''
        
        view                   = [1] * ndim
        view[self.output_axis] = -1
        return values.reshape(view)
