# @file quant_sima_block.py
# @brief 量子化した transformer のブロック: INT8 の残差の上の SimA の attention と 1x1 conv の MLP で、整数の計算だけで動く
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver     import Granularity, QuantAdd, QuantConv2d, Quantizer, QuantParams, QuantSimaAttention, RangeScaler, SharedScaler, build_format
from quaver.ops import OpSet, mode_ops
from .types     import CnnVitSetting



class QuantSimaBlock(nn.Module):
    '''
    #* 量子化した SimA のブロック。attention と MLP は、それぞれの出力を INT8 の残差に足す。
    #? scale の共有（Kaneko と Meng の Fig. 7。標準はこちら）: ブロックの入力、attention と MLP の出力、2 つの和は、1 つの scale を共有する（ブロックごとに SharedScaler が 1 つ）。
    #? 枝の出力はその scale に requantize し（デバイスではカーネルにまとめる）、どの足し算も飽和付きのただの INT8 の足し算になる。次のブロックは自分の scale から始める。
    #? 共有しないとき（shared_residual=False）は、どの足し算も、枝の INT32 の出力と残差を、自分の INT8 の scale に requantize する。
    #? branch_format があるとき（4 bit のモデル）は、attention と MLP の入力で、INT8 の残差をその形式に requantize する。層の名前は SimaBlock と同じ。
    #! QuantSimaAttention は、非対称の setting でも自分の対称の量子化器を持つ。
    '''
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int, hidden: int, setting: CnnVitSetting) -> None:
        '''
        #* attention、MLP の 2 つの conv、INT8 の足し算 2 つ、枝の入力の量子化器、残差のストリームの共有の scale を作る。
        '''
        
        # attention と MLP の入力: INT8 の残差をそのまま使うか、静的な scale で枝の形式に requantize する
        super().__init__()
        branch               = setting.branch_format
        self.attention_input = None if branch is None else Quantizer(build_format(branch), RangeScaler(symmetric=setting.act_symmetric), Granularity(), batch_axis=0)
        self.mlp_input       = None if branch is None else Quantizer(build_format(branch), RangeScaler(symmetric=setting.act_symmetric), Granularity(), batch_axis=0)
        
        # 残差のストリーム: ブロックで 1 つの共有の scale（ブロックの入力の量子化器、枝の出力、足し算が使う）か、なし
        self.shared   = SharedScaler(symmetric=setting.act_symmetric) if setting.shared_residual else None
        self.residual = None if self.shared is None else Quantizer(build_format("int8"), self.shared, Granularity(), batch_axis=0)
        output        = None if self.shared is None else "int8"
        
        # SimA の attention（projection は、共有の scale の INT8 か、INT32 の累積で終わる）と、INT8 の残差の足し算
        self.attention = QuantSimaAttention(channels, heads, key_dim, value_dim, weight_format=setting.weight_format, act_format=setting.sima_format, project_format=setting.project_format, output_format=output, output_scaler=self.shared)
        self.add1      = QuantAdd("int8", output_symmetric=setting.act_symmetric, output_scaler=self.shared)
        
        # MLP: 1x1 conv、ReLU、活性化の形式
        self.expand = QuantConv2d(channels, hidden, 1, weight_format=setting.weight_format, output_format=setting.act_format, activation="relu", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        
        # 共有の scale の INT8（または INT32 の累積）に出す 1x1 conv と、INT8 の残差の足し算
        self.shrink       = QuantConv2d(hidden, channels, 1, weight_format=setting.weight_format, output_format=output, weight_symmetric=setting.weight_symmetric, output_scaler=self.shared)
        self.add2         = QuantAdd("int8", output_symmetric=setting.act_symmetric, output_scaler=self.shared)
        self.integer_mode = False
        self.float_mode   = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* 今のモード（fake の値、整数の code、float の値）でブロックを動かす: x + attention(x)、次に x + MLP(x) で、どちらも INT8 に出す。
        '''
        
        # 共有の scale にしたブロックの入力（Fig. 7 の最初の量子化器）。scaler は前のパスの範囲を忘れる
        ops = mode_ops(self.integer_mode, self.float_mode)
        if self.shared is not None:
            self.shared.begin_forward()
            x, x_params = self._quant_sima_block_branch(self.residual, x, x_params, ops)
        
        # attention: 枝の入力、SimA、INT8 の残差の足し算
        a, a_params = self._quant_sima_block_branch(self.attention_input, x, x_params, ops)
        y, y_params = self.attention(a, a_params)
        x, x_params = self.add1(y, y_params, x, x_params)
        
        # MLP: 枝の入力、2 つの conv、INT8 の残差の足し算
        m, m_params = self._quant_sima_block_branch(self.mlp_input, x, x_params, ops)
        h, h_params = self.expand(m, m_params)
        y, y_params = self.shrink(h, h_params)
        x, x_params = self.add2(y, y_params, x, x_params)
        
        # 共有の scale を、このパスで一番広い範囲に近づける（学習中か calibration 中だけ）
        if self.shared is not None:
            self.shared.end_forward()
        return x, x_params
    
    def _quant_sima_block_branch(self, quantizer: Quantizer | None, x: torch.Tensor, x_params: QuantParams, ops: OpSet) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* 入力を量子化器の scale に requantize する（静的な M0 と n）。量子化器がないときは、入力をそのまま返す。
        #? forward() が、ブロックの入力（共有の scale）と、attention と MLP の入力（枝の形式）に使う。
        '''
        
        if quantizer is None:
            return x, x_params
        params = quantizer.params(x) if self.integer_mode else quantizer.measure(x)
        return ops.requantize(x, x_params, params, "mul16_shift"), params
