from .basic_block       import BasicBlock
from .build             import MODEL_REGISTRY, build_model
from .nvfp4_basic_block import Nvfp4BasicBlock
from .nvfp4_resnet8     import Nvfp4Resnet8
from .quant_basic_block import QuantBasicBlock
from .quant_resnet8     import QuantResnet8
from .resnet8           import Resnet8
from .types             import Resnet8Setting

'''
model/
├── __init__.py
├── basic_block.py
│   └── BasicBlock
│       └── forward()
├── build.py
│   └── build_model()
├── nvfp4_basic_block.py
│   └── Nvfp4BasicBlock
│       └── forward()
├── nvfp4_resnet8.py
│   └── Nvfp4Resnet8
│       └── forward()
├── quant_basic_block.py
│   └── QuantBasicBlock
│       └── forward()
├── quant_resnet8.py
│   └── QuantResnet8
│       └── forward()
├── resnet8.py
│   └── Resnet8
│       └── forward()
└── types.py
    └── Resnet8Setting
'''
