# @file types.py
# @brief 量子化した CNN + ViT の設定
# @author Kohei Yamaguchi, Claude

from dataclasses import dataclass



@dataclass(frozen=True)
class CnnVitSetting:
    '''
    #* 量子化した CNN + ViT 1 つの、量子化の設定を持つ。
    #? build.py の MODEL_REGISTRY がモデルの名前ごとに設定を 1 つ与え、QuantCnnVit がそれをもとに層を作る。
    #! act_format: ReLU のあとの出力（conv1、conv2、MLP）。branch_format: attention と MLP の入力（None のときは INT8 の残差をそのまま使う）。
    #! sima_format: Q、K、attention map、V。project_format: projection の入力（符号付き。論文と同じく attention x V のあとに ReLU6 を置かない）。非対称の設定は INT の形式でしか使えない。
    #! shared_residual: 各ブロックの中の残差のストリームで 1 つの scale を使う（Kaneko と Meng の Fig. 7 の scale の共有）。False のときは、足し算ごとに自分の scale を持つ。
    '''
    
    weight_format   : str
    act_format      : str
    branch_format   : str | None
    sima_format     : str
    project_format  : str
    weight_symmetric: bool = True
    act_symmetric   : bool = True
    shared_residual : bool = True
