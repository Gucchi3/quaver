# @file device.py
# @brief デバイスの integer の手順を torch の op（torch.ops.quaver.*）として登録し、torch.export が各手順を 1 つのノードとして残すようにする
# @author Kohei Yamaguchi, Claude

##########################################################
'''
デバイスの op を足すとき
1. op の隣に xxx_device(...) を書く: デバイスが動かす integer の手順で、ただのテンソル、int、名前だけを使う（QuantParams は使わない）。
2. それに @device_define("名前") を付ける。
3. op の integer の関数から torch.ops.quaver.名前(...) として呼び、torch.export に 1 つのノードとして見せる。
'''
##########################################################

from collections.abc import Callable
from typing          import Any, Literal, get_origin

import torch
import types

# quaver の op のライブラリ。CompositeExplicitAutograd は、どのデバイスでも、torch.export の fake のテンソルでも、同じ Python のコードを動かす
DEVICE_LIBRARY = torch.library.Library("quaver", "DEF")



def device_define(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    '''
    #* 関数を op torch.ops.quaver.<名前> として登録し、関数をそのまま返すデコレータを返す。
    #? スキーマは型ヒントから作る。torch の op はただの文字列を受け取るので、Literal の名前（RoundMode など）は str になる。
    #! op は torch.ops.quaver.<名前> を通して呼ぶ。関数そのものを呼んでも動くが、そのときは torch.export に中の手順が見える。
    '''
    
    def register(function: Callable[..., Any]) -> Callable[..., Any]:
        '''
        #* function をその名前で登録する。Literal のヒントは str として読む。
        '''
        
        plain                 = types.FunctionType(function.__code__, function.__globals__, function.__name__, function.__defaults__, function.__closure__)
        plain.__annotations__ = {key: str if get_origin(value) is Literal else value for key, value in function.__annotations__.items()}
        DEVICE_LIBRARY.define(name + torch.library.infer_schema(plain, mutates_args=()))
        DEVICE_LIBRARY.impl(name, function, "CompositeExplicitAutograd")
        return function
    
    return register
