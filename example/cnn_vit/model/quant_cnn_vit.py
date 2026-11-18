# @file quant_cnn_vit.py
# @brief quaver のサンプルで作った、整数の計算だけで動く量子化した CNN + ViT（pooling はない）
# @author Kohei Yamaguchi, Claude

from torch import nn

import torch

from quaver            import QuantConv2d, QuantLinear, QuantParams, build_format
from .quant_sima_block import QuantSimaBlock
from .types            import CnnVitSetting



class QuantCnnVit(nn.Module):
    '''
    #* 量子化した CNN + ViT。入力は UINT8（scale 1/255）、最初の conv と分類器の重みは INT8、ほかの層は setting の形式で、残差のストリームは INT8 にする。
    #? 層の名前は CnnVit と同じなので、畳み込んだ FP32 の重みをそのまま読み込める。switch_integer_mode() でモデル全体を integer mode にする。pooling はない。
    #! どちらのモードでも実数の logits を返す。integer mode では、分類器の正確な INT32 の累積から求める。
    '''
    
    input_scale: torch.Tensor
    
    def __init__(self, setting: CnnVitSetting) -> None:
        '''
        #* stem、ブロック、分類器を作る。
        #? 最初の conv と分類器の重みは、いつも INT8 にする。stem の最後の conv（embed）は、ReLU なしで残差のストリーム（トークン）を出す。
        #? scale を共有するときは、embed が INT32 の累積を出し、最初のブロックがそれを共有の scale に量子化する（丸めは 1 回）。共有しないときは、embed が INT8 を出す。
        '''
        
        # stem: conv1（重みは INT8）と conv2 は ReLU のあと活性化の形式にし、embed は残差に出す（最初のブロックの共有の scale のために INT32、または INT8）
        super().__init__()
        residual   = None if setting.shared_residual else "int8"
        self.conv1 = QuantConv2d(3 , 16, 3, 1, 1, weight_format="int8"               , output_format=setting.act_format, activation="relu", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        self.conv2 = QuantConv2d(16, 32, 3, 2, 1, weight_format=setting.weight_format, output_format=setting.act_format, activation="relu", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        self.embed = QuantConv2d(32, 64, 3, 2, 1, weight_format=setting.weight_format, output_format=residual          , activation="none", weight_symmetric=setting.weight_symmetric, output_symmetric=setting.act_symmetric)
        
        # 64 個のトークンに SimA のブロック 2 つ、すべてのトークンを読む分類器（重みは INT8、logits は INT32）
        self.blocks       = nn.ModuleList([QuantSimaBlock(64, 4, 16, 16, 128, setting) for _ in range(2)])
        self.classifier   = QuantLinear(64 * 8 * 8, 10, weight_format="int8", weight_symmetric=setting.weight_symmetric)
        self.integer_mode = False
        self.register_buffer("input_scale", torch.tensor(1 / 255))
        
        return None
    
    def forward(self, images: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, QuantParams]:
        '''
        #* training mode では uint8 の画像の logits を返し、integer mode では logits の INT32 の code とそのパラメータを返す。
        #? integer mode はデバイスと同じく code を出すので、書き出したモデルは分類器で終わる。params_values() でそれを logits に変える。
        '''
        
        # 入力は scale 1/255 の uint8 の code: training mode では fake の値、integer mode では code
        params = QuantParams(self.input_scale, torch.zeros_like(self.input_scale), build_format("uint8"))
        x      = images.long() if self.integer_mode else images.float() * self.input_scale
        
        # stem、SimA のブロック、平らにしたトークンを読む分類器
        x, params = self.conv1(x, params)
        x, params = self.conv2(x, params)
        x, params = self.embed(x, params)
        for block in self.blocks:
            x, params = block(x, params)
        x, params = self.classifier(x.flatten(1), params)
        return (x, params) if self.integer_mode else x
