from .export      import model_export
from .formats     import build_format
from .model_tools import calibrate_model, fold_batch_norms, freeze_scalers, lsq_clamp_steps, lsq_param_groups, precision_disable_tf32, precision_set_math, switch_integer_mode, switch_quantizers
from .quantizers  import Quantizer, QuantizerSpec, build_quantizer
from .samples     import QuantAdd, QuantAvgPool2d, QuantBlockConv2d, QuantBlockLinear, QuantConv2d, QuantLinear, QuantSimaAttention
from .scales      import BlockScaler, DynamicScaler, Granularity, LsqScaler, QuantParams, RangeScaler, SharedScaler, params_codes, params_values

'''
quaver/
├── __init__.py    # 基本の API: サンプルの層、量子化器、モデル全体の道具（ほかの公開の名前は、それぞれのフォルダにある）
├── formats/       # 数値形式（INT、FP、UFP）、正確な丸め、デバイスの op の登録（torch.ops.quaver）
├── scales/        # scale の選び方: dynamic、max、percentile、LSQ、共有、ブロックの scale（NVFP4、整数）
├── quantizers/    # Quantizer（fake quantization）、LSQ、QuantizerSpec から作る関数
├── ops/           # training mode と integer mode の量子化した op（integer の手順はデバイスの op）
├── model_tools/   # モデル全体の道具: 集める、切り替える、止める、calibration、LSQ、畳み込み、TF32、同期
├── samples/       # サンプルの層: Conv2d、Linear、ブロックごとの scale の Conv2d と Linear（NVFP4 など）、SimA の attention、残差の足し算、平均の pooling
└── export/        # integer mode のモデルをデバイス用に書き出す: model.json と model.bin（numpy が要る）
'''
