# @file build.py
# @brief QuantizerSpec から量子化器を作る
# @author Kohei Yamaguchi, Claude

##########################################################
'''
scaler を足すとき
1. scaler を返す build_xxx_scaler(spec, scale_count) を書く。
2. SCALER_REGISTRY に "名前: build_xxx_scaler" を足す。
3. QuantizerSpec.scaler をその名前にする。
'''
##########################################################

from collections.abc import Callable

from ..formats  import build_format
from ..scales   import BlockScaler, DynamicScaler, Granularity, LsqScaler, RangeScaler, SharedScaler
from .quantizer import Quantizer
from .types     import QuantizerSpec, Scaler



def build_dynamic_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* 毎回、今のテンソルを測る scaler を作る。
    '''
    
    return DynamicScaler(percentile=None, symmetric=spec.symmetric)


def build_max_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* 最大を移動平均で追う scaler を作る。
    '''
    
    return RangeScaler(percentile=None, momentum=spec.momentum, scale_count=scale_count, symmetric=spec.symmetric, freeze_step=spec.freeze_step)


def build_percentile_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* percentile（標準は 99.9）を移動平均で追う scaler を作る。
    '''
    
    return RangeScaler(percentile=spec.percentile, momentum=spec.momentum, scale_count=scale_count, symmetric=spec.symmetric, freeze_step=spec.freeze_step)


def build_lsq_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* step を学習する scaler（LSQ。非対称なら LSQ+）を作る。
    '''
    
    return LsqScaler(scale_count=scale_count, symmetric=spec.symmetric, init=spec.lsq_init)


def build_shared_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* 複数の量子化器で共有できる scaler を作る。
    #! 呼ぶたびに新しいものを作る。共有するときは、SharedScaler を 1 つ作り、それを各 Quantizer に渡す。
    '''
    
    return SharedScaler(percentile=None, momentum=spec.momentum, symmetric=spec.symmetric, freeze_step=spec.freeze_step)


def build_nvfp4_scaler(spec: QuantizerSpec, scale_count: int) -> Scaler:
    '''
    #* NVFP4 の 2 段の scale を持つ scaler を作る: ブロックごとの UE4M3 の scale と、毎回測るテンソルの scale。
    '''
    
    return BlockScaler("ue4m3")


SCALER_REGISTRY: dict[str, Callable[[QuantizerSpec, int], Scaler]] = {
    "dynamic"   : build_dynamic_scaler,
    "max"       : build_max_scaler,
    "percentile": build_percentile_scaler,
    "lsq"       : build_lsq_scaler,
    "shared"    : build_shared_scaler,
    "nvfp4"     : build_nvfp4_scaler,
}


def build_quantizer(spec: QuantizerSpec, scale_count: int = 1) -> Quantizer:
    '''
    #* QuantizerSpec から Quantizer を作る。
    #? scale_count は、max、percentile、lsq の scale の数（たとえばチャネルの数）。
    #! 新しい scaler は SCALER_REGISTRY に足す。
    '''
    
    # scaler を作る関数を探す
    if spec.scaler not in SCALER_REGISTRY:
        raise KeyError(f"Unknown scaler {spec.scaler!r}. Available: {', '.join(SCALER_REGISTRY)}")
    
    # 部品を組み合わせる
    scaler      = SCALER_REGISTRY[spec.scaler](spec, scale_count)
    granularity = Granularity(spec.granularity, spec.axis, spec.block_size)
    return Quantizer(build_format(spec.format), scaler, granularity, spec.round_mode, spec.pow2, spec.batch_axis)
