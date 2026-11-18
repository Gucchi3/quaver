# @file build.py
# @brief 名前で選ぶ、量子化した CNN + ViT のモデル
# @author Kohei Yamaguchi, Claude

##########################################################
'''
モデルを足すとき
1. CnnVitSetting を書く: 重みの形式、ReLU のあとの活性化の形式、attention と MLP の入力の形式、SimA の形式、projection の入力の形式。
2. MODEL_REGISTRY に "名前: setting" を足す。
3. main.py --models 名前 で動かす。
'''
##########################################################

from .quant_cnn_vit import QuantCnnVit
from .types         import CnnVitSetting

# 最初の conv と分類器の重みは INT8 で、残差のストリームも INT8 にする。4 bit のモデルはそれ以外をすべて 4 bit にする。attention と MLP の入力で、残差を 4 bit に
# requantize する。対称のモデルは ReLU のあとに符号なしの形式を使い、非対称のモデルは zero point のある符号付きの形式を使う（INT だけ）。
# SimA はどのモデルでも自分の対称の量子化器を持ち、projection の入力は符号付きにする（論文と同じく、attention x V のあとに ReLU6 を置かない）。
MODEL_REGISTRY: dict[str, CnnVitSetting] = {
    "int8"     : CnnVitSetting(weight_format="int8", act_format="uint8", branch_format=None  , sima_format="int8", project_format="int8", weight_symmetric=True , act_symmetric=True ),
    "int8_asym": CnnVitSetting(weight_format="int8", act_format="int8" , branch_format=None  , sima_format="int8", project_format="int8", weight_symmetric=False, act_symmetric=False),
    "int4"     : CnnVitSetting(weight_format="int4", act_format="uint4", branch_format="int4", sima_format="int4", project_format="int4", weight_symmetric=True , act_symmetric=True ),
    "int4_asym": CnnVitSetting(weight_format="int4", act_format="int4" , branch_format="int4", sima_format="int4", project_format="int4", weight_symmetric=False, act_symmetric=False),
    "fp4"      : CnnVitSetting(weight_format="fp4" , act_format="ufp4" , branch_format="fp4" , sima_format="fp4" , project_format="fp4" , weight_symmetric=True , act_symmetric=True ),
}



def build_model(name: str) -> QuantCnnVit:
    '''
    #* MODEL_REGISTRY にある名前の、量子化した CNN + ViT を返す。
    #? main.py がこれで各モデルを作り、そこに畳み込んだ FP32 の重みを読み込む。
    '''
    
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODEL_REGISTRY)}")
    return QuantCnnVit(MODEL_REGISTRY[name])
