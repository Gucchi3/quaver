# @file model_export.py
# @brief integer mode のモデルをデバイス用に書き出す: model.json（層、設定、形式、各データの場所）と model.bin（すべてのデータを詰めたもので、そのまま PSRAM に置ける）
# @author Kohei Yamaguchi, Claude

##########################################################
'''
書き出しにデバイスの op を足すとき
1. op を device_define()（formats/device.py）で書き、op の integer の関数から呼ぶ。そこでパラメータが分かるときは、出力に params_note() で印を付ける。
2. EXPORT_FORMATS に "名前: {テンソルの引数か出力の番号: その形式の引数、または形式の名前}" を足し、code に形式が付くようにする。
3. EXPORT_ROLES に "名前: {テンソルの引数: 役割}" を足し、データがどの層でも同じ名前（weight、multiplier など）になるようにする。
層のクラスを足すとき
1. integer_mode を持たせ、integer mode をデバイスの op で動かす。設定は、名前が EXPORT_SETTINGS にあるときに JSON に入る。
2. Write its role names and settings in docs/設計.md (9章, "実機との約束").
'''
##########################################################

from collections.abc import Callable
from pathlib         import Path
from torch           import nn
from torch.fx.node   import Node, map_arg
from typing          import Any, cast

import inspect
import json
import math
import operator
import re
import torch

from ..formats      import build_format
from ..ops          import ACCUMULATOR_FORMAT, WIDE_ACCUMULATOR_FORMAT, accumulate_bits
from ..scales       import params_named
from .model_outputs import ModelOutputs
from .pack          import pack_codes
from .types         import ExportLayout

# デバイスの op ごとの code の形式: テンソルの引数は名前で、出力は番号で、形式の名前を持つ引数か、形式の名前そのものを書く
# （zero point はその code と同じ形式、multiplier は 16 bit、shift は int8 に入る: -30〜94）
EXPORT_FORMATS: dict[str, dict[str | int, str]] = {
    "quantize"           : {"zero_point": "number_format", 0: "number_format"},
    "accumulate_conv2d"  : {"q_x": "x_format", "x_zero_point": "x_format", "x_block": "x_scale_format", "q_w": "w_format", "w_zero_point": "w_format", "w_block": "w_scale_format", 0: "out_format"},
    "accumulate_linear"  : {"q_x": "x_format", "x_zero_point": "x_format", "x_block": "x_scale_format", "q_w": "w_format", "w_zero_point": "w_format", "w_block": "w_scale_format", 0: "out_format"},
    "accumulate_matmul"  : {"q_a": "a_format", "a_zero_point": "a_format", "a_block": "a_scale_format", "q_b": "b_format", "b_zero_point": "b_format", "b_block": "b_scale_format", 0: "out_format"},
    "activation"         : {},
    "requantize"         : {"multiplier": "uint16", "shift": "int8", "out_zero_point": "out_format", 0: "out_format"},
    "add"                : {"a_multiplier": "uint16", "a_shift": "int8", "b_multiplier": "uint16", "b_shift": "int8", "out_zero_point": "out_format", 0: "out_format"},
    "pool_avg2d"         : {"q_x": "number_format", "zero_point": "number_format", 0: "number_format"},
    "normalize_l1"       : {"multiplier": "uint16", "shift": "int8", 0: "out_format", 1: "uint21"},
    "normalize_l1_scores": {"multiplier": "uint16", "shift": "int8", "out_zero_point": "out_format", 0: "out_format"},
    "block_quantize"     : {"scale_multiplier": "uint16", "scale_shift": "int8", "value_multiplier": "uint16", "value_shift": "int8", 0: "out_format", 1: "scale_format"},
    "dynamic_scale"      : {},
    "fixed_point"        : {0: "uint16", 1: "int8"},
}

# op の定数の引数ごとの役割: その層の中でのデータの名前（同じクラスのどのインスタンスでも同じ）。ほかの引数は "op_引数" という名前になる
EXPORT_ROLES: dict[str, dict[str, str]] = {
    "quantize"           : {"scale": "scale", "zero_point": "zero_point"},
    "accumulate_conv2d"  : {"x_zero_point": "input_zero_point", "x_block": "input_block_codes", "q_w": "weight", "w_zero_point": "weight_zero_point", "w_block": "weight_block_codes"},
    "accumulate_linear"  : {"x_zero_point": "input_zero_point", "x_block": "input_block_codes", "q_w": "weight", "w_zero_point": "weight_zero_point", "w_block": "weight_block_codes"},
    "accumulate_matmul"  : {"a_zero_point": "a_zero_point", "a_block": "a_block_codes", "q_b": "b", "b_zero_point": "b_zero_point", "b_block": "b_block_codes"},
    "activation"         : {"zero_point": "clamp_min", "six": "clamp_max"},
    "requantize"         : {"zero_point": "accumulator_zero_point", "multiplier": "multiplier", "shift": "shift", "out_zero_point": "output_zero_point"},
    "add"                : {"a_zero_point": "a_zero_point", "a_multiplier": "a_multiplier", "a_shift": "a_shift", "b_zero_point": "b_zero_point", "b_multiplier": "b_multiplier", "b_shift": "b_shift", "out_zero_point": "output_zero_point"},
    "pool_avg2d"         : {"zero_point": "input_zero_point"},
    "normalize_l1"       : {"zero_point": "neq_zero_point", "multiplier": "neq_multiplier", "shift": "neq_shift"},
    "normalize_l1_scores": {"multiplier": "scores_multiplier", "shift": "scores_shift", "out_zero_point": "scores_zero_point"},
    "block_quantize"     : {"zero_point": "block_zero_point", "scale_multiplier": "block_scale_multiplier", "scale_shift": "block_scale_shift", "value_multiplier": "block_value_multiplier", "value_shift": "block_value_shift"},
    "aten.add"           : {"other": "bias"},
}

# 1 つ目より後ろの出力の役割（op と番号ごと）: それを出す層の出力は "output_役割" になる
EXPORT_OUTPUTS: dict[str, dict[int, str]] = {
    "block_quantize": {1: "block_codes"},
    "normalize_l1"  : {1: "peak"},
}

# module が持っているときに、JSON に入る層の設定
EXPORT_SETTINGS = ("stride", "padding", "dilation", "groups", "activation", "style", "round_mode", "heads", "key_dim", "value_dim", "neq_scope", "output_block", "kernel_size")

# レイアウトの名前（2 次元では hwc と chw と呼ぶ）と、空間の軸が 1、2、3 個のときの軸の名前
EXPORT_LAYOUTS = {"channels_last": "channels_last", "channels_first": "channels_first", "hwc": "channels_last", "chw": "channels_first"}
EXPORT_SPATIAL = {1: ["k"], 2: ["h", "w"], 3: ["d", "h", "w"]}

# 値の torch の型の名前
EXPORT_TYPES = {
    torch.bool   : "bool"   ,
    torch.uint8  : "uint8"  ,
    torch.int8   : "int8"   ,
    torch.int16  : "int16"  ,
    torch.int32  : "int32"  ,
    torch.int64  : "int64"  ,
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}



def model_export(model: nn.Module, inputs: tuple[torch.Tensor, ...], folder: str | Path, alignment: int = 4, layout: dict[str, ExportLayout] | None = None, json_values: bool = True) -> dict[str, Any]:
    '''
    #* integer mode のモデルを、デバイス用に folder/model.json と folder/model.bin に書き出し、JSON を dict で返す。
    #? model.json: 層を順に並べる（ブロックは "layers"、ほかの層は "ops" を持つ）。層の入力と出力を名前で書き、形式と形、設定、各データの役割、形式、encoding、形、レイアウト、offset、大きさを書く。
    #? model.bin: すべてのデータをすき間なく詰め、それぞれを alignment バイトの倍数から始める。offset は PSRAM の中の番地で、ファイルの初めから数える。
    #? torch.export が forward(*inputs) を記録し、入力によらない値はここで計算する（重みの code、バイアス、M0 と n、zero point、ブロックの code）。層の中の op は詳しい情報として残し、参照の読み手がそれを動かす。
    #? layout は、層のパスかクラスの名前を "channels_last"（標準。2 次元では "hwc"）か "channels_first"（"chw"）に対応させる。子の層はブロックのレイアウトを受け継ぐ。channels_last では、conv の重みを (out, spatial..., in) の順に置く。
    #? channels_last の活性化を平らにして Linear に入れるときは、その重みの列を (spatial..., channel) の順に並べ替えるので、デバイスの結果は同じになる。
    #? json_values=True のときは、すべてのデータの数も 10 進で書く（"values"。model.bin と同じ順で、stored_shape の形に入れ子にする）。INT の code は整数、FP の code は実数の値（fp4 なら 1.5）、float は float で書く。
    #! 先にモデルを integer mode（switch_integer_mode）と eval() にしておく。forward() はテンソル、組 (codes, QuantParams)、またはそれらの tuple か list を返してよい。numpy が要る（"export" の extra）。
    '''
    
    # この入口で確かめる: integer mode、eval()、multiplier と shift での requantize、alignment、レイアウトの名前
    if not all(getattr(module, "integer_mode") for module in model.modules() if hasattr(module, "integer_mode")):
        raise ValueError("Put the model in the integer mode first: switch_integer_mode(model, True).")
    if any(module.training for module in model.modules()):
        raise ValueError("Put the model in eval() first, so that the scales stay fixed and become constants.")
    if any(getattr(module, "style", None) == "exact" for module in model.modules()):
        raise ValueError("The style 'exact' keeps real ratios, but the device needs M0 and n. Use mul16_shift or shift_only.")
    if alignment < 1:
        raise ValueError(f"alignment must be 1 or more, not {alignment}.")
    if any(value not in EXPORT_LAYOUTS for value in (layout or {}).values()):
        raise ValueError(f"Unknown layout in {layout}. Available: {', '.join(EXPORT_LAYOUTS)}")
    
    # 勾配なしでグラフを記録する（層の中の no_grad() は何もしない）。ModelOutputs は、forward() が返す組の code を、そのパラメータの印を付けて返す
    # torch.export は cuDNN の古い TF32 のフラグを読む。PyTorch 2.14 はこれを conv と RNN がどちらも "tf32" のときしか許さないので、precision_disable_tf32() のあとでは読めない。
    # フラグはすぐあとで元に戻す。integer mode は float64 と int64 で計算し、記録のときは実際の計算をしないので、ここで TF32 は何も変えない
    wrapped    = ModelOutputs(model)
    cudnn      = [getattr(torch.backends.cudnn, kind) for kind in ("conv", "rnn")]
    precisions = [backend.fp32_precision for backend in cudnn]
    try:
        for backend in cudnn:
            backend.fp32_precision = "tf32"
        with torch.no_grad():
            program = torch.export.export(wrapped, inputs)
    finally:
        for backend, precision in zip(cudnn, precisions):
            backend.fp32_precision = precision
    
    # パラメータ、buffer、定数の値と、forward() の引数の名前
    signature = program.graph_signature
    tensors   = {**program.state_dict, **program.constants}
    sources   = {**signature.inputs_to_parameters, **signature.inputs_to_buffers, **signature.inputs_to_lifted_tensor_constants}
    known     = {name: tensors[target] for name, target in sources.items()}
    nodes     = [node for node in program.graph.nodes if node.op == "call_function"]
    arguments = [name for name, parameter in inspect.signature(model.forward).parameters.items() if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
    
    # 入力は forward() の引数の名前のままにする
    rename: dict[str, str] = {str(name): str(argument) for name, argument in zip(signature.user_inputs, arguments)} if len(arguments) >= len(signature.user_inputs) else {}
    
    # 入力がすべて分かっているノードを畳み込む: scale、M0 と n、重みの code とそのブロックの scale、バイアスの code
    for node in nodes:
        if all(value.name in known for value in node.all_input_nodes):
            args, kwargs     = map_arg((node.args, node.kwargs), lambda value: known[value.name])
            known[node.name] = cast(Callable[..., Any], node.target)(*cast(tuple, args), **cast(dict, kwargs))
    
    # 印と、値を変えない型の変換は op ではない。それぞれ、読んでいる値の代わりになる
    notes = {node.name: node for node in nodes if node.target is torch.ops.quaver.note.default}
    alias = {}
    for node in nodes:
        source  = node.all_input_nodes[0] if node.all_input_nodes else None
        before  = source.meta.get("val") if source is not None else None
        after   = node.meta.get("val")
        if source is None or not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor) or source.name in known or node.name in known:
            continue
        integer = not before.dtype.is_floating_point and not after.dtype.is_floating_point and torch.bool not in (before.dtype, after.dtype)
        widens  = integer and torch.iinfo(after.dtype).min <= torch.iinfo(before.dtype).min and torch.iinfo(before.dtype).max <= torch.iinfo(after.dtype).max
        keeps   = getattr(node.target, "__name__", "").split(".")[0] in ("to", "_to_copy", "detach", "clone", "alias") and len(node.all_input_nodes) == 1 and (before.dtype == after.dtype or widens) and tuple(before.shape) == tuple(after.shape)
        if node.name in notes or keeps:
            alias[node.name] = alias.get(source.name, rename.get(source.name, source.name))
    
    # 出力に要る値を、出力から逆にたどって集める（分かっている値は、そこでデータになる）
    results: list[Node] = []
    map_arg(program.graph.output_node().args[0], results.append)
    if any(node.name in known for node in results):
        raise ValueError("An output does not depend on the inputs, so it has no node to export.")
    
    # 分かっていないノードを逆にたどる（印や残した型の変換は、それが読む値へ進む）
    needed  = set()
    pending = list(results)
    while pending:
        node = pending.pop()
        if node.name in alias:
            pending.append(node.all_input_nodes[0])
        elif node.name not in needed and node.name not in known:
            needed.add(node.name)
            pending.extend(node.all_input_nodes)
    
    # すべての値の情報: 形式と、分かっているときは scale と zero point（グラフの値なら名前で書く）。入力は型を書く
    metadata: dict[str, dict[str, Any]] = {str(rename.get(node.name, node.name)): {"format": EXPORT_TYPES[node.meta["val"].dtype]} for node in program.graph.nodes if node.name in signature.user_inputs}
    
    # 要る op ごとに記録を 1 つ作り、それが動く層のつながり（integer_mode を持つ module を、外から内へ、モデルより下だけ）を書く
    constants    : dict[str, dict] = {}
    biases       : dict[str, str]  = {}
    labels       : dict[str, str]  = {}
    named_outputs: dict[str, str]  = {}
    records      : list[dict]      = []
    for node in nodes:
        
        # 印は、印を付けた値に、形式、scale、zero point を書く。定数は数で、グラフの値は名前で書く（定数の印は畳み込まれて消える）
        if node.name in notes and node.name not in alias:
            continue
        if node.name in notes:
            target                          = alias[node.name]
            _, scale, zero_point, form, key = node.args
            
            # 形式（wide の印は、書き出しが int32 で書いた和の int32 を残す: activation_integer() は累積に wide の形式の印を付ける）、次に scale と zero point
            narrowed                                  = form == WIDE_ACCUMULATOR_FORMAT.name and metadata.get(target, {}).get("format") == ACCUMULATOR_FORMAT.name
            metadata.setdefault(target, {})["format"] = ACCUMULATOR_FORMAT.name if narrowed else cast(str, form)
            for name, value in ((key, scale), ("zero_point", zero_point)):
                if isinstance(value, Node) and value.name in known:
                    metadata[target][cast(str, name)] = torch.as_tensor(known[value.name]).tolist()
                elif isinstance(value, Node) and value.name in needed:
                    metadata[target][cast(str, name)] = {"value": alias.get(value.name, value.name)}
            continue
        if node.name not in needed or node.name in alias or node.target is operator.getitem:
            continue
        if not hasattr(node.target, "_schema"):
            raise ValueError(f"Cannot export {node.target} ({node.name}): only torch ops and quaver device ops can be exported.")
        
        # スキーマ、デバイスの op の形式の名前、名前で書いた引数、層
        schema       = getattr(node.target, "_schema")
        domain, name = schema.name.split("::")
        spec         = EXPORT_FORMATS.get(name, {}) if domain == "quaver" else {}
        bound        = {argument.name: node.args[index] if index < len(node.args) else node.kwargs.get(argument.name) for index, argument in enumerate(schema.arguments)}
        stack        = node.meta.get("nn_module_stack", {})
        chain        = [(key, place.removeprefix("model."), type(wrapped.get_submodule(place)).__name__) for key, (place, _) in stack.items() if place.startswith("model.") and hasattr(wrapped.get_submodule(place), "integer_mode")]
        names        = []
        roles        = []
        attributes   = {}
        
        # 累積は、一番悪い場合が 32 bit に入るときに int32 と書く（accumulate_bits()。そのすぐあとに足すバイアスの |code| の最大も含める）。バイアスは、その累積の形式にする
        if name.startswith("accumulate_") and domain == "quaver":
            adds  = [user for user in node.users if user.target is torch.ops.aten.add.Tensor and isinstance(user.args[1], Node) and user.args[1].name in known]
            peak  = max((int(torch.as_tensor(known[cast(Node, user.args[1]).name]).abs().max()) for user in adds), default=0)
            sides = [params_named(torch.zeros(()), cast(str, bound[f"{side}_format"]), torch.zeros(()) if bound[f"{side}_scale_format"] else None, cast(str, bound[f"{side}_scale_format"])) for side in ("ab" if name == "accumulate_matmul" else "xw")]
            shape = cast(Node, bound["q_a" if name == "accumulate_matmul" else "q_w"]).meta["val"].shape
            count = math.prod(shape[1:]) if name == "accumulate_conv2d" else shape[-1]
            if accumulate_bits(sides[0], sides[1], count, peak) <= 32:
                bound["out_format"] = ACCUMULATOR_FORMAT.name
            biases.update({user.name: cast(str, bound["out_format"]) for user in adds})
        
        # ほかの引数は属性にする: list、数、文字列、名前で書いた torch の型（device は書かない）
        for argument in schema.arguments:
            value = bound[argument.name]
            if "Tensor" not in str(argument.type):
                value = known[value.name] if isinstance(value, Node) else value
                value = [*value] if isinstance(value, (list, tuple)) else str(value) if isinstance(value, (torch.dtype, torch.memory_format, torch.layout)) else value
                if value is not None and not isinstance(value, torch.device):
                    attributes[argument.name] = value
                continue
            
            # テンソルの引数は入力にする（None なら None）。テンソルの list は長さも書く
            if isinstance(value, (list, tuple)):
                attributes[f"{argument.name}_count"] = len(value)
            for position, item in enumerate(value if isinstance(value, (list, tuple)) else [value]):
                source = item.name if isinstance(item, Node) else f"{node.name}_{argument.name}_{position}"
                if item is None or (isinstance(item, Node) and item.name not in known) or source in labels:
                    names.append(None if item is None else labels.get(source, alias.get(source, rename.get(source, source))))
                    roles.append(EXPORT_ROLES.get(name if domain == "quaver" else f"{domain}.{name}", {}).get(argument.name, ""))
                    continue
                
                # 新しいデータ。最初に使った所で名前を付ける: 層のパスと引数の役割（同じ層で同じ役割を 2 回使うと _2 を付ける）
                role           = EXPORT_ROLES.get(name if domain == "quaver" else f"{domain}.{name}", {}).get(argument.name, f"{name}_{argument.name}")
                label          = f"{chain[-1][1]}.{role}" if chain else role
                label          = label if label not in constants else next(f"{label}_{count}" for count in range(2, len(constants) + 3) if f"{label}_{count}" not in constants)
                labels[source] = label
                tensor         = torch.as_tensor(known[item.name] if isinstance(item, Node) else item).detach().cpu()
                format_name    = (cast(str, bound[spec[argument.name]]) if spec[argument.name] in bound else spec[argument.name]) if argument.name in spec else ""
                if not format_name:
                    format_name = biases.get(node.name) or (EXPORT_TYPES[tensor.dtype] if tensor.dtype.is_floating_point else "int32" if tensor.numel() == 0 or (-2**31 <= int(tensor.min()) and int(tensor.max()) < 2**31) else "int64")
                
                # データ（2 回目に使った役割は _2 を付けたままにする）
                constants[label] = {"role": label.removeprefix(f"{chain[-1][1]}.") if chain else label, "tensor": tensor, "format": format_name, "path": chain[-1][1] if chain else ""}
                names.append(label)
                roles.append(role)
        
        # 出力: 値 1 つか、getitem が tuple から取り出す値（何も使わない部分は None）
        if isinstance(node.meta["val"], (list, tuple)):
            parts   = {user.args[1]: user for user in node.users if user.target is operator.getitem}
            outputs = [parts[index].name if index in parts else None for index in range(len(node.meta["val"]))]
            shapes  = list(node.meta["val"])
        else:
            outputs = [node.name]
            shapes  = [node.meta["val"]]
        
        # 各出力の形式: op から決めるか、同じ型の値である最初の入力から受け継ぐ（形が変わらないか scale が 1 つの数のときは、scale と zero point も受け継ぐ）
        first = next((item.meta.get("val") for item in node.all_input_nodes if item.name not in known)                                                    , None)
        kept  = next((metadata.get(alias.get(item.name, rename.get(item.name, item.name)), {}) for item in node.all_input_nodes if item.name not in known), {}  )
        for index, (output, value) in enumerate(zip(outputs, shapes)):
            if not output or not isinstance(value, torch.Tensor):
                continue
            if index in spec:
                metadata[output] = {"format": cast(str, bound[spec[index]]) if spec[index] in bound else spec[index]}
                continue
            if not (isinstance(first, torch.Tensor) and first.dtype == value.dtype and kept):
                metadata[output] = {"format": EXPORT_TYPES[value.dtype]}
                continue
            metadata[output] = {key: item for key, item in kept.items() if key == "format" or tuple(first.shape) == tuple(value.shape) or not isinstance(item, list)}
        
        # op の手順（torch の op が標準でない overload のときは、それも書く）
        step = {"op": f"{domain}.{name}", "inputs": names, "outputs": outputs, "attributes": attributes}
        if getattr(node.target, "_overloadname", "default") != "default":
            step["overload"] = getattr(node.target, "_overloadname")
        records.append({"step": step, "inputs": names, "roles": roles, "outputs": outputs, "chain": chain, "groups": []})
        named_outputs.update({cast(str, outputs[index]): role for index, role in EXPORT_OUTPUTS.get(name if domain == "quaver" else "", {}).items() if outputs[index]})
    
    # モデルの出力（組の code には、そのパラメータの印が付いている）
    outputs_needed = [alias.get(node.name, node.name) for node in results]
    
    # 記録を層の呼び出しにまとめる: 1 つの層の 1 回の呼び出しの中の、続いた記録を 1 つのグループにし、グループは module と同じように入れ子にする
    root  : dict           = {"id": 0, "items": [], "overload": "", "layout": EXPORT_LAYOUTS[(layout or {}).get("", "channels_last")]}
    groups: list[dict]     = [root]
    calls : dict[str, int] = {}
    for record in records:
        parent = root
        
        # 層のつながりを下へたどり、層の新しい呼び出しが始まる所で新しいグループを始める。層のレイアウトは、パス、クラス、ブロックのどれかから決める
        for key, place, kind in record["chain"]:
            last   = parent["items"][-1] if parent["items"] else None
            if last is None or last.get("key") != key:
                calls[place] = calls.get(place, 0) + 1
                chosen       = (layout or {}).get(place, (layout or {}).get(kind))
                last         = {"id": len(groups), "key": key, "kind": kind, "overload": place if calls[place] == 1 else f"{place}@{calls[place]}", "module": wrapped.get_submodule(f"model.{place}"), "items": [],
                                "layout": EXPORT_LAYOUTS[chosen] if chosen else parent["layout"]}
                groups.append(last)
                parent["items"].append(last)
            parent = last
            record["groups"].append(parent["id"])
        parent["items"].append(record)
    
    # 各グループの入力（中で読み、外で作るもの）と出力（中で作り、外かモデルが読むもの）
    for group in groups[1:]:
        inside           = [record for record in records if group["id"] in record["groups"]]
        made             = {name for record in inside for name in record["outputs"] if name}
        read             = {name for record in records if group["id"] not in record["groups"] for name in record["inputs"]} | set(outputs_needed)
        group["inputs"]  = list(dict.fromkeys(name for record in inside for name in record["inputs"] if name and name not in made and name not in constants))
        group["outputs"] = list(dict.fromkeys(name for record in inside for name in record["outputs"] if name in read)                                      )
    
    # 層が出す値の読みやすい名前: "パス.output"（次は "パス.output_役割" か "パス.output_2"）。外の層から先に付ける
    edges: dict[str, str] = {}
    for group in groups[1:]:
        for index, name in enumerate(group["outputs"]):
            edges.setdefault(name, f"{group['overload']}.output" + ("" if index == 0 else f"_{named_outputs[name]}" if name in named_outputs else f"_{index + 1}"))
    for record in records:
        record["step"]["inputs"]  = [edges.get(name, name) if name else None for name in record["step"]["inputs"]]
        record["step"]["outputs"] = [edges.get(name, name) if name else None for name in record["step"]["outputs"]]
        record["inputs"]          = record["step"]["inputs"]
    for group in groups[1:]:
        group["inputs"], group["outputs"] = [edges.get(name, name) for name in group["inputs"]], [edges.get(name, name) for name in group["outputs"]]
    
    # この名前で書いたモデルの出力と、それぞれの値を作る記録
    outputs_needed = [edges.get(name, name) for name in outputs_needed]
    makers         = {name: record for record in records for name in record["outputs"] if name}
    
    # 要る値すべての形式、形、パラメータを、最後の名前で書く
    values: dict[str, dict[str, Any]] = {}
    for node in program.graph.nodes:
        value = node.meta.get("val")
        name  = str(rename.get(node.name, node.name))
        if node.name in needed and isinstance(value, torch.Tensor):
            if value.dtype not in EXPORT_TYPES:
                raise ValueError(f"The value {node.name} has the type {value.dtype}, which the export does not handle.")
            values[edges.get(name, name)] = {"name": edges.get(name, name), **metadata.get(name, {}), "shape": list(value.shape)}
    
    # 層の呼び出しごとの設定: module の設定、形、レイアウト、最初の入力と出力の形式、出力の code の範囲
    for group in groups[1:]:
        module   = group["module"]
        settings = {key: getattr(module, key) for key in EXPORT_SETTINGS if isinstance(getattr(module, key, None), (bool, int, str, tuple, list))}
        weight   = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim >= 3:
            settings.update(out_channels=weight.shape[0], in_channels=weight.shape[1], kernel_size=list(weight.shape[2:]))
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            settings.update(out_features=weight.shape[0], in_features=weight.shape[1])
        if isinstance(getattr(getattr(module, "blocks", None), "block_size", None), int):
            settings["block_size"] = getattr(module, "blocks").block_size
        settings.update(layout=group["layout"])
        
        # 最初の入力、最初の出力、重みの形式と、出力の code の範囲
        sources = {"input_format": values.get(group["inputs"][0], {}), "output_format": values.get(group["outputs"][0], {}), "weight_format": constants.get(f"{group['overload']}.weight", {})}
        settings.update({key: source["format"] for key, source in sources.items() if source.get("format")})
        if settings.get("output_format"):
            named = build_format(str(settings["output_format"]))
            settings.update(output_min=round(named.min_value / named.step_min), output_max=round(named.max_value / named.step_min))
        
        # tuple は list にし、層が画像を扱うか（カーネルを持つか）を書く
        group["settings"] = {key: [*value] if isinstance(value, (tuple, list)) else value for key, value in settings.items()}
        group["spatial"]  = "kernel_size" in group["settings"]
    for group in reversed(groups[1:]):
        group["spatial"] = group["spatial"] or any(item.get("spatial") for item in group["items"])
    
    # 層の入力と出力での画像のレイアウト（rank 4 以上と、カーネルを持つ層での rank 3）: channels_last は (batch, spatial..., channel) の順に置く
    for group in groups[1:]:
        for name in [*group["inputs"], *group["outputs"]]:
            rank    = len(values[name]["shape"]) if name in values else 0
            spatial = EXPORT_SPATIAL.get(rank - 2, [f"s{index}" for index in range(rank - 2)]) if rank >= 4 or (rank == 3 and group["spatial"]) else None
            if spatial is not None and "layout" not in values[name]:
                producer               = makers[name]["groups"][-1] if name in makers and makers[name]["groups"] else group["id"]
                chosen                 = groups[producer]["layout"]
                values[name]["layout"] = chosen
                values[name]["axes"]   = ["batch", *spatial, "channel"] if chosen == "channels_last" else ["batch", "channel", *spatial]
    
    # データ: それぞれを置く順（レイアウト、ブロックは 1 回だけ、平らにした入力のための列の並べ替え）にし、alignment を守ってすき間なく詰める
    blob    = bytearray()
    entries = {group["overload"]: [] for group in groups}
    for label, constant in constants.items():
        group  = next((group for group in groups if group["overload"] == constant["path"]), root)
        tensor = constant["tensor"]
        item   = {"role": constant["role"], "name": label, "format": constant["format"], "shape": list(tensor.shape)}
        
        # 空間の軸を持つ重み: channels_last は (out, spatial..., in) の順にする。ブロックの code は、入力の軸のブロックごとに 1 回だけ置く
        if constant["role"] in ("weight", "weight_block_codes") and tensor.ndim >= 3:
            size    = group["settings"].get("block_size") if constant["role"] == "weight_block_codes" else None
            if size and tensor.shape[1] % size == 0 and tensor.shape[1] > 1:
                tensor         = tensor.unflatten(1, (-1, size))[:, :, 0]
                item["blocks"] = {"axis": 1, "size": size}
            spatial = EXPORT_SPATIAL.get(tensor.ndim - 2, [f"s{index}" for index in range(tensor.ndim - 2)])
            inner   = "in_block" if "blocks" in item else "in"
            if group["layout"] == "channels_last":
                tensor         = tensor.permute(0, *range(2, tensor.ndim), 1)
                item["layout"] = "channels_last"
                item["axes"]   = ["out", *spatial, inner]
            else:
                item["layout"] = "channels_first"
                item["axes"]   = ["out", inner, *spatial]
        
        # Linear の重み: (out, in)。層の前で channels_last の画像を平らにするときは、列を (spatial..., channel) の順に並べ替える。ただし、空間の軸の大きさがすべて 1 なら並べ替えない
        elif constant["role"] in ("weight", "weight_block_codes") and tensor.ndim == 2:
            maker  = makers.get(group["inputs"][0]) if group["inputs"] else None
            source = values.get(cast(str, maker["inputs"][0])) if maker and maker["step"]["op"] == "aten.flatten" else None
            if source and len(source["shape"]) >= 3 and source.get("layout") == "channels_last" and any(size > 1 for size in source["shape"][2:]):
                if constant["role"] == "weight_block_codes" or "block_size" in group["settings"]:
                    raise ValueError(f"{group['overload']}: a block-scaled Linear after a flatten of channels_last images would mix its blocks. Give the layer before it the layout channels_first.")
                channels, *spatial_sizes = source["shape"][1:]
                tensor                   = tensor.unflatten(1, (channels, *spatial_sizes)).permute(0, *range(2, 2 + len(spatial_sizes)), 1).flatten(1)
                spatial                  = EXPORT_SPATIAL.get(len(spatial_sizes), [f"s{index}" for index in range(len(spatial_sizes))])
                item["reorder"]          = {"from": ["channel", *spatial], "to": [*spatial, "channel"], "shape": [channels, *spatial_sizes]}
            
            # Linear の重みは、どちらのレイアウトでも (out, in)
            item["layout"] = "row_major"
            item["axes"]   = ["out", "in"]
        
        # ほかのデータ（バイアス、zero point、M0 と n、clamp、scale）: 大きさ 1 の軸は書かない
        else:
            kept           = [index for index, size in enumerate(tensor.shape) if size != 1]
            tensor         = tensor.reshape([tensor.shape[index] for index in kept])
            item["layout"] = "flat"
            item["axes"]   = ["out" if constant["role"].startswith("weight") else "channel"] if len(kept) == 1 else [f"axis{index}" for index in kept]
        
        # 次の alignment の offset に詰める
        packed, encoding = pack_codes(tensor.contiguous(), constant["format"])
        blob.extend(bytes(-len(blob) % alignment))
        item.update(stored_shape=list(tensor.shape), **encoding, offset=len(blob), size=len(packed))
        blob.extend(packed)
        entries[group["overload"]].append(item)
        
        # 同じ数を、置く順に 10 進で書く: INT の code は整数、FP の code は形式の実数の値、float はそのまま
        if json_values:
            named          = None if constant["format"] in ("float32", "float64") else build_format(constant["format"])
            item["values"] = tensor.double().tolist() if named is None else (tensor.double() * named.step_min).tolist() if named.kind == "fp" else tensor.long().tolist()
    
    # 各層の JSON: クラス、形式と形付きの入力と出力、設定、データ、子（"layers"。ブロックの op を子の間に並べる）か op
    for group in reversed(groups):
        steps = [item["step"] if "step" in item else item["json"] for item in group["items"]]
        if group is root:
            continue
        
        # ブロックは子と、その間の op を "layers" に持ち、子のない層は op を "ops" に持つ
        group["json"] = {"name": group["overload"], "class": group["kind"], "inputs": [values[name] for name in group["inputs"]], "outputs": [values[name] for name in group["outputs"]], "settings": group["settings"],
                         "data": entries[group["overload"]], ("layers" if any("json" in item for item in group["items"]) else "ops"): steps}
    
    # モデル: ファイルの形式、入力と出力、層の外のデータ、層、すべての値
    exported = {
        "quaver"     : 1,
        "alignment"  : alignment,
        "byte_order" : "little",
        "json_values": json_values,
        "size"       : len(blob),
        "codes"      : "INT codes are two's complement, FP codes bit patterns (sign, exponent, mantissa). 2-bit and 4-bit codes fill each byte from the low bits up. An offset is the address in the file. values: the same numbers in decimal.",
        "inputs"     : [values[rename.get(str(name), str(name))] for name in signature.user_inputs if rename.get(str(name), str(name)) in values],
        "outputs"    : [values[str(name)] for name in outputs_needed],
        "data"       : entries[""],
        "layers"     : [item["step"] if "step" in item else item["json"] for item in root["items"]],
        "values"     : values,
    }
    
    # JSON の文字列: 数か文字列だけの list は 1 行にし（"values" の 1 行が 1 行になる）、ほかは字下げする。写しには、その list の代わりに印を入れておく
    marked = json.loads(json.dumps(exported))
    rows   = []
    stack  = [marked]
    while stack:
        node = stack.pop()
        for key, child in list(node.items() if isinstance(node, dict) else enumerate(node)):
            if isinstance(child, list) and not any(isinstance(value, (dict, list)) for value in child):
                node[key] = f"\0{len(rows)}"
                rows.append(child)
            elif isinstance(child, (dict, list)):
                stack.append(child)
    text   = re.sub(r'"\\u0000(\d+)"', lambda found: json.dumps(rows[int(found.group(1))]), json.dumps(marked, indent=1))
    
    # 2 つのファイル
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.json").write_bytes(text.encode("utf-8"))
    (target / "model.bin").write_bytes(bytes(blob))
    return exported
