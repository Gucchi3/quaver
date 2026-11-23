# @file build.py
# @brief 名前で選ぶ、量子化した ResNet-8 のモデル
# @author Kohei Yamaguchi, Claude

##########################################################
'''
モデルを足すとき
1. Resnet8Setting を書く: 重みと活性化の形式、それぞれが対称かどうか、scale がブロックごとか（NVFP4）。
2. MODEL_REGISTRY に "名前: setting" を足す。
3. main.py --models 名前 で動かす。
'''
##########################################################

from .nvfp4_resnet8 import Nvfp4Resnet8
from .quant_resnet8 import QuantResnet8
from .types         import Resnet8Setting

# 対称のモデルは zero point をすべて 0 にするので、ReLU のあとに符号なしの形式を使う。
# 非対称のモデルはすべての重みと活性化に zero point を持たせるので、ReLU のあとに符号付きの形式を使う（0 は一番小さい code になる）。FP の形式は対称だけ。
# nvfp4 は、block conv の重みを NVFP4 にし、その入力をブロックごとの scale の UFP4 にする（16 個ずつのブロックで、scale は UE4M3）。stem、分類器、残差は INT8 / UINT8 のまま。
MODEL_REGISTRY: dict[str, Resnet8Setting] = {
    "int8"     : Resnet8Setting(weight_format="int8", act_format="uint8", weight_symmetric=True , act_symmetric=True , block_scaled=False),
    "int8_asym": Resnet8Setting(weight_format="int8", act_format="int8" , weight_symmetric=False, act_symmetric=False, block_scaled=False),
    "int4"     : Resnet8Setting(weight_format="int4", act_format="uint4", weight_symmetric=True , act_symmetric=True , block_scaled=False),
    "int4_asym": Resnet8Setting(weight_format="int4", act_format="int4" , weight_symmetric=False, act_symmetric=False, block_scaled=False),
    "fp4"      : Resnet8Setting(weight_format="fp4" , act_format="ufp4" , weight_symmetric=True , act_symmetric=True , block_scaled=False),
    "nvfp4"    : Resnet8Setting(weight_format="fp4" , act_format="ufp4" , weight_symmetric=True , act_symmetric=True , block_scaled=True ),
}



def build_model(name: str) -> QuantResnet8 | Nvfp4Resnet8:
    '''
    #* MODEL_REGISTRY にある名前の、量子化した ResNet-8 を返す。ブロックごとの scale の setting なら Nvfp4Resnet8、それ以外なら QuantResnet8 にする。
    #? main.py がこれで各モデルを作り、そこに畳み込んだ FP32 の重みを読み込む。
    '''
    
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODEL_REGISTRY)}")
    setting = MODEL_REGISTRY[name]
    return Nvfp4Resnet8(setting) if setting.block_scaled else QuantResnet8(setting)
