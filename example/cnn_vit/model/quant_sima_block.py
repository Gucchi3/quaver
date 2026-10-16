# @file quant_sima_block.py
# @brief Quantized transformer block: SimA attention and an MLP of 1x1 convs on an INT8 residual, with integer math only
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver     import Granularity, QuantAdd, QuantConv2d, Quantizer, QuantParams, QuantSimaAttention, RangeScaler, SharedScaler, build_format
from quaver.ops import OpSet, mode_ops
from .types     import CnnVitSetting



class QuantSimaBlock(nn.Module):
    '''
    #* Quantized SimA block: the attention and the MLP each add their output to the INT8 residual.
    #? Scale sharing (Kaneko and Meng, Fig. 7; the default): the block input, the outputs of the attention and the MLP, and both sums share one scale (one SharedScaler for the block).
    #? The outputs of the branches are requantized to it (fused into the kernels on the device), and each add is a plain INT8 add with saturation. The next block starts its own scale.
    #? Without it (shared_residual=False), each add requantizes the INT32 output of its branch and the residual to its own INT8 scale.
    #? With branch_format (4-bit models), the INT8 residual is requantized to it at the input of the attention and of the MLP. The layer names are those of SimaBlock.
    #! QuantSimaAttention keeps its own symmetric quantizers, also in the asymmetric settings.
    '''
    
    def __init__(self, channels: int, heads: int, key_dim: int, value_dim: int, hidden: int, setting: CnnVitSetting) -> None:
        '''
        #* Make the attention, the two convs of the MLP, the two INT8 adds, the quantizers of the branch inputs, and the shared scale of the residual stream.
        '''
        
        # The inputs of the attention and the MLP: the INT8 residual as it is, or requantized to the branch format with a static scale
        super().__init__()
        branch               = setting.branch_format
        self.attention_input = None if branch is None else Quantizer(build_format(branch), RangeScaler(symmetric=setting.act_symmetric), Granularity(), batch_axis=0)
        self.mlp_input       = None if branch is None else Quantizer(build_format(branch), RangeScaler(symmetric=setting.act_symmetric), Granularity(), batch_axis=0)
        
        # The residual stream: one shared scale for the block (the quantizer of the block input, the outputs of the branches, and the adds), or none
        self.shared   = SharedScaler(symmetric=setting.act_symmetric) if setting.shared_residual else None
        self.residual = None if self.shared is None else Quantizer(build_format("int8"), self.shared, Granularity(), batch_axis=0)
        output        = None if self.shared is None else "int8"
        
        # SimA attention (its projection ends in INT8 on the shared scale, or in the INT32 accumulator), and the INT8 residual add
        self.attention = QuantSimaAttention(channels, heads, key_dim, value_dim, weight_format=setting.weight_format, act_format=setting.sima_format, project_format=setting.project_format, output_format=output, output_scaler=self.shared)
        self.add1      = QuantAdd("int8", output_symmetric=setting.act_symmetric, output_scaler=self.shared)
        
        # MLP: a 1x1 conv with ReLU into the activation format
        self.expand = QuantConv2d(channels, hidden, 1, weight_format=setting.weight_format, output_format=setting.act_format, activation="relu", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        
        # A 1x1 conv into INT8 on the shared scale (or the INT32 accumulator), and the INT8 residual add
        self.shrink       = QuantConv2d(hidden, channels, 1, weight_format=setting.weight_format, output_format=output, weight_symmetric=setting.weight_symmetric, output_scaler=self.shared)
        self.add2         = QuantAdd("int8", output_symmetric=setting.act_symmetric, output_scaler=self.shared)
        self.integer_mode = False
        self.float_mode   = False
        
        return None
    
    def forward(self, x: torch.Tensor, x_params: QuantParams) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Run the block in the current mode (fake values, integer codes, or float values): x + attention(x), then x + MLP(x), both into INT8.
        '''
        
        # The block input on the shared scale (the first quantizer of Fig. 7). The scaler forgets the ranges of the last pass
        ops = mode_ops(self.integer_mode, self.float_mode)
        if self.shared is not None:
            self.shared.begin_forward()
            x, x_params = self._quant_sima_block_branch(self.residual, x, x_params, ops)
        
        # Attention: the branch input, SimA, and the INT8 residual add
        a, a_params = self._quant_sima_block_branch(self.attention_input, x, x_params, ops)
        y, y_params = self.attention(a, a_params)
        x, x_params = self.add1(y, y_params, x, x_params)
        
        # MLP: the branch input, the two convs, and the INT8 residual add
        m, m_params = self._quant_sima_block_branch(self.mlp_input, x, x_params, ops)
        h, h_params = self.expand(m, m_params)
        y, y_params = self.shrink(h, h_params)
        x, x_params = self.add2(y, y_params, x, x_params)
        
        # The shared scale moves toward the widest range of this pass (only while training or calibrating)
        if self.shared is not None:
            self.shared.end_forward()
        return x, x_params
    
    def _quant_sima_block_branch(self, quantizer: Quantizer | None, x: torch.Tensor, x_params: QuantParams, ops: OpSet) -> tuple[torch.Tensor, QuantParams]:
        '''
        #* Requantize the input to the scale of the quantizer (static M0 and n), or return it as it is when there is no quantizer.
        #? forward() uses this for the block input (the shared scale) and for the inputs of the attention and of the MLP (the branch format).
        '''
        
        if quantizer is None:
            return x, x_params
        params = quantizer.params(x) if self.integer_mode else quantizer.measure(x)
        return ops.requantize(x, x_params, params, "mul16_shift"), params
