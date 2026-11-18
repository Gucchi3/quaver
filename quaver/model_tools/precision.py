# @file precision.py
# @brief float の設定: TF32 を止め、training mode の計算の float の型を選ぶ
# @author Kohei Yamaguchi, Claude

import torch

from ..scales import MATH_DTYPES, MathDtype

MATH_TYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}



def precision_disable_tf32() -> None:
    '''
    #* 畳み込みと行列積の TF32 を止める。
    #? cuDNN は標準で float32 の畳み込みに TF32 を使い、そうすると INT32 の累積を正確に取り戻せない。
    #! これはプロセス全体の設定である。RTX 3060 では step の時間が約 10% 増えた。
    '''
    
    # torch の型ヒントには cudnn.conv がないので、getattr() で取り出す
    torch.backends.cuda.matmul.fp32_precision            = "ieee"
    getattr(torch.backends.cudnn, "conv").fp32_precision = "ieee"
    
    return None


def precision_set_math(train: MathDtype = "float64", test: MathDtype = "float64") -> None:
    '''
    #* training mode の計算の float の型を選ぶ: train は勾配があるとき（学習の step）、test は勾配がないとき（torch.no_grad() の中のテスト）に使う。
    #? float32 にすると学習が約 1.6 倍速くなる（ResNet-8、CUDA graph あり）。テストを float64 にすれば、デバイスの code を正確に得られる。integer mode はいつも float64 を使う。
    #! float32 では、まれな所で training mode がデバイスと 1 code ずれることがある。CUDA graph を記録する前に呼ぶ（graph は型を覚えている）。
    '''
    
    # 名前はこの入口で確かめる
    for name in (train, test):
        if name not in MATH_TYPES:
            raise KeyError(f"Unknown math type {name!r}. Available: {', '.join(MATH_TYPES)}")
    
    # 勾配がある step と、勾配がない step の型を決める
    MATH_DTYPES["train"] = MATH_TYPES[train]
    MATH_DTYPES["test"]  = MATH_TYPES[test]
    
    return None
