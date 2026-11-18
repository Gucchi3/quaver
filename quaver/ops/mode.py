# @file mode.py
# @brief すべての op の、float 版、training 版、integer 版を選ぶ
# @author Kohei Yamaguchi, Claude

from .activation import activation_fake, activation_float, activation_integer
from .add        import add_fake, add_float, add_integer
from .block      import block_quantize_fake, block_quantize_float, block_quantize_integer
from .matmul     import matmul_fake, matmul_float, matmul_integer
from .normalize  import normalize_l1_fake, normalize_l1_float, normalize_l1_integer, normalize_l1_scores_fake, normalize_l1_scores_float, normalize_l1_scores_integer
from .pool       import pool_avg2d_fake, pool_avg2d_float, pool_avg2d_integer
from .requantize import requantize_fake, requantize_float, requantize_integer
from .types      import OpSet

FLOAT_OPS   = OpSet(requantize=requantize_float  , add=add_float  , pool_avg2d=pool_avg2d_float  , matmul=matmul_float  , normalize_l1=normalize_l1_float  , normalize_l1_scores=normalize_l1_scores_float  , activation=activation_float  ,
                    block_quantize=block_quantize_float  )
FAKE_OPS    = OpSet(requantize=requantize_fake   , add=add_fake   , pool_avg2d=pool_avg2d_fake   , matmul=matmul_fake   , normalize_l1=normalize_l1_fake   , normalize_l1_scores=normalize_l1_scores_fake   , activation=activation_fake   ,
                    block_quantize=block_quantize_fake   )
INTEGER_OPS = OpSet(requantize=requantize_integer, add=add_integer, pool_avg2d=pool_avg2d_integer, matmul=matmul_integer, normalize_l1=normalize_l1_integer, normalize_l1_scores=normalize_l1_scores_integer, activation=activation_integer,
                    block_quantize=block_quantize_integer)



def mode_ops(integer_mode: bool, float_mode: bool = False) -> OpSet:
    '''
    #* integer mode、training mode、float のモードのどれかの op を返す。
    #? 層は forward をこの op で 1 回だけ書く。training mode と integer mode は同じ整数の計算をする。float のモード（calibration）はただの float の op を動かす。
    #! integer_mode は float_mode より優先する。
    '''
    
    if integer_mode:
        return INTEGER_OPS
    return FLOAT_OPS if float_mode else FAKE_OPS
