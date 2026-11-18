# @file quant_sima_attention.py
# @brief 整数の計算だけで動く、量子化した SimA の attention（softmax の代わりに L1 正規化）のサンプル
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from ..formats     import RoundMode, build_format
from ..ops         import ActivationKind, NeqScope, OpSet, build_requantize, mode_ops
from ..quantizers  import Quantizer, Scaler
from ..scales      import Granularity, QuantParams, RangeScaler
from .quant_conv2d import QuantConv2d



class QuantSimaAttention(nn.Module):
    '''
    #* softmax のない SimA の attention: Q と K をトークンごとに L1 正規化して NEQ（Kaneko と Meng）で量子化し、(Q K^T) V を求める。デバイスではすべて整数で計算する。
    #? E_AI_deit の SimA の ViT と同じく、(B, C, H, W) の上で Q、K、V、projection に 1x1 conv を使う（その 1 / sqrt(key_dim) は scale に入れる）。
    #? 論文と同じく、標準では projection の前に活性化はない。activation="relu6" で E_AI_deit の ReLU6 を足す。
    #! NEQ は動的（標準ではサンプルと head ごとに scale を 1 つ。論文の Algorithm 2）で、デバイスは attention map の M0 と n を整数で求める。
    '''
    
    activation: ActivationKind
    neq_scope : NeqScope
    round_mode: RoundMode | None
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int, weight_format: str = "int4", act_format: str = "int4", project_format: str = "int4", activation: ActivationKind = "none", neq_scope: NeqScope = "tensor",
                 output_format: str | None = None, output_scaler: Scaler | None = None, style: str = "mul16_shift", round_mode: RoundMode | None = None) -> None:
        '''
        #* 4 つの 1x1 conv と、attention map と projection の入力の量子化器を作る。
        #? act_format は Q、K、attention map、V の形式。activation（"none"、"relu"、"relu6"）は attention x V にかける。neq_scope: "tensor"（論文の設定）か "token"。
        #? output_format があるときは、projection をそれに requantize する（デバイスではカーネルにまとめる）。scale を残差と共有するときは output_scaler を使う（scale の共有）。None のときは INT32 の累積を返す。
        #! 活性化がないときは符号付きの project_format を、ReLU か ReLU6 のあとでは符号なしの形式を使う（符号なしの形式は、ReLU と同じく負の値を切る）。
        #! 整数の計算が正確なままになるように、style には multiplier と shift（mul16_shift か shift_only）が、act_format には 8 bit 以下の code が要る。
        '''
        
        # この入口で確かめる: 動的な scale には M0 と n が要り、積は正確なままでなければならない
        number_format = build_format(act_format)
        if build_requantize(style).multiplier_max == 0:
            raise ValueError(f"SimA needs a style with a multiplier and a shift (mul16_shift or shift_only), not {style!r}: its scales change while running.")
        if number_format.max_value / number_format.step_min > 255:
            raise ValueError(f"act_format must have codes of 8 bits or less (INT8, INT4, FP4, ...), not {act_format!r}.")
        
        # 設定
        super().__init__()
        self.heads         = heads
        self.key_dim       = key_dim
        self.value_dim     = value_dim
        self.weight_format = weight_format
        self.act_format    = number_format
        self.activation    = activation
        self.neq_scope     = neq_scope
        self.style         = style
        self.round_mode    = round_mode
        self.integer_mode  = False
        self.float_mode    = False
        
        # Q、K、V、projection の 1x1 conv
        self.query   = self._sima_conv(channels         , heads * key_dim  , None         , None         )
        self.key     = self._sima_conv(channels         , heads * key_dim  , None         , None         )
        self.value   = self._sima_conv(channels         , heads * value_dim, act_format   , None         )
        self.project = self._sima_conv(heads * value_dim, channels         , output_format, output_scaler)
        
        # attention map と projection の入力の scale は最大に従う（軸 0 は batch）
        self.attention_quantizer = Quantizer(self.act_format             , RangeScaler(), Granularity(), round_mode, batch_axis=0)
        self.project_quantizer   = Quantizer(build_format(project_format), RangeScaler(), Granularity(), round_mode, batch_axis=0)
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* 今のモード（fake の値、整数の code、float の値）で SimA の attention を動かす。
        #? projection の出力とそのパラメータを返す。そのまま残差の足し算に使える: INT32 の累積か、output_format の code。
        #! x_params の scale は、テンソル全体で 1 つでなければならない。
        '''
        
        # Q と K: INT32 の累積に出す 1x1 conv、head への分割、NEQ（サンプルと head ごと、またはトークンごとに scale を 1 つ）
        ops                     = mode_ops(self.integer_mode, self.float_mode)
        batch, _, height, width = x.shape
        query, query_params     = self._sima_tokens(self.query, x, x_params, ops)
        key, key_params         = self._sima_tokens(self.key  , x, x_params, ops)
        
        # attention map: 正確な Q K^T を NEQ の整数の最大で requantize する。1 / sqrt(key_dim) は multiplier の静的な部分に入れる
        factor               = self.key_dim**-0.5
        key_t_params         = QuantParams(key_params.scale.transpose(-1, -2), key_params.zero_point.transpose(-1, -2), key_params.number_format, peak=None if key_params.peak is None else key_params.peak.transpose(-1, -2))
        scores, score_params = ops.matmul(query, query_params, key.transpose(-1, -2), key_t_params)
        attention_params     = self.attention_quantizer.params(scores) if self.integer_mode else self.attention_quantizer.measure(scores * factor)
        attention            = ops.normalize_l1_scores(scores, score_params, query_params, key_t_params, attention_params, factor, self.style)
        
        # value: 活性化の形式に出す 1x1 conv と、head への分割
        value, value_params = self.value(x, x_params)
        value               = value.reshape(batch, self.heads, self.value_dim, -1).transpose(-1, -2)
        
        # 出力: 正確な attention x V、累積への活性化（標準ではなし）、そして projection の入力への requantize（静的な M0 と n）
        output, output_params = ops.matmul(attention, attention_params, value, value_params)
        output                = ops.activation(output, output_params, self.activation)
        project_params        = self.project_quantizer.params(output) if self.integer_mode else self.project_quantizer.measure(output)
        output                = ops.requantize(output, output_params, project_params, self.style)
        output                = output.transpose(-1, -2).reshape(batch, self.heads * self.value_dim, height, width)
        return self.project(output, project_params)
    
    def _sima_conv(self, in_channels: int, out_channels: int, output_format: str | None, output_scaler: Scaler | None) -> QuantConv2d:
        '''
        #* 共通の重みの形式、style、丸めで、1x1 conv を 1 つ作る。
        #? __init__() が 4 つの conv に使う。
        '''
        
        return QuantConv2d(in_channels, out_channels, 1, weight_format=self.weight_format, output_format=output_format, style=self.style, round_mode=self.round_mode, output_scaler=output_scaler)
    
    def _sima_tokens(self, conv: QuantConv2d, x: torch.Tensor, x_params: QuantParams, ops: OpSet) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Q か K の conv を累積まで動かし、head に分けて (B, heads, tokens, key_dim) にし、NEQ をかける。
        #? forward() が Q と K に使う。累積はチャネルごとに scale を 1 つ持つので、scale も同じように head に分ける。
        '''
        
        acc, acc_params  = conv(x, x_params)
        acc, scale, zero = [part.reshape(part.shape[0], self.heads, self.key_dim, -1).transpose(-1, -2) for part in (acc, acc_params.scale, acc_params.zero_point)]
        return ops.normalize_l1(acc, QuantParams(scale, zero, acc_params.number_format), self.act_format, self.style, self.neq_scope)
