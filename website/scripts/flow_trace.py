# @file flow_trace.py
# @brief ガイドのモデルが、各モードで Quaver のどの関数をどの順に呼ぶかを記録する。API リファレンスの流れのページ用
# @author Kohei Yamaguchi, Claude

from pathlib import Path

import ast
import json
import sys
import torch

from quaver   import calibrate_model, precision_disable_tf32, switch_integer_mode
from tiny_net import TinyNet

ROOT    = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "quaver"
GUIDE   = Path(__file__).resolve().parent
OUTPUT  = ROOT / "website" / "src" / "data" / "flow.json"



def flow_trace_main() -> None:
    '''
    #* profiler の下で各モードでモデルを動かし、Quaver への呼び出しを木として残して、サイト用の JSON に書く。
    #? 木には、quaver/ の関数と、このスクリプトの隣のモデルのファイルの関数の呼び出しを残す。各 docstring の #* の行が、その呼び出しが何のためかを表す。
    #! コードを変えたら、リポジトリの一番上から動かす: uv run python website/scripts/flow_trace.py
    '''
    
    # すべての関数とメソッドの #* の行を、ファイルと修飾名ごとに集める
    notes: dict[tuple[str, str], str] = {}
    for path in [*PACKAGE.rglob("*.py"), *GUIDE.glob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            items = [(f"{node.name}.{item.name}", item) for item in node.body if isinstance(item, ast.FunctionDef)] if isinstance(node, ast.ClassDef) else []
            for name, item in items:
                lines = (ast.get_docstring(item) or "").splitlines()
                notes.setdefault((str(path), name), next((line.strip()[3:] for line in lines if line.strip().startswith("#*")), ""))
            if isinstance(node, ast.FunctionDef):
                lines = (ast.get_docstring(node) or "").splitlines()
                notes.setdefault((str(path), node.name), next((line.strip()[3:] for line in lines if line.strip().startswith("#*")), ""))
    
    # モデルを作り、batch で scale を決め、module に名前を付ける（conv1、depthwise2 など）
    precision_disable_tf32()
    torch.manual_seed(0)
    model  = TinyNet()
    images = torch.randn(2, 3, 32, 32)
    calibrate_model(model, [images])
    model(images)
    names  = {id(module): name or "model" for name, module in model.named_modules()}
    
    # 記録するモード: キー、題、見せるもの、動かし方
    def run_training() -> None:
        '''
        #* QAT の 1 step: train() で forward と backward。
        '''
        
        model.train()
        model(images).sum().backward()
        return None
    
    def run_test() -> None:
        '''
        #* テスト: eval() で、勾配なしに forward。
        '''
        
        model.eval()
        with torch.no_grad():
            model(images)
        return None
    
    def run_integer() -> None:
        '''
        #* integer mode: code で forward し、training mode に戻す。
        '''
        
        model.eval()
        switch_integer_mode(model, True)
        with torch.no_grad():
            model(images)
        switch_integer_mode(model, False)
        return None
    
    def run_calibration() -> None:
        '''
        #* calibration: float のモードで calibrate_model()。
        '''
        
        calibrate_model(model, [images])
        return None
    
    modes = [
        ("training"   , "Training step", "One forward pass of QAT, in train(): the scalers update, and every step that rounds runs the integer math of the device.", run_training   ),
        ("test"       , "Test"         , "The same model in eval() and torch.no_grad(): the scales stay fixed."                                                  , run_test       ),
        ("integer"    , "Integer mode" , "The model on integer codes, with integer math only, as on the device."                                                 , run_integer    ),
        ("calibration", "Calibration"  , "calibrate_model(): the float mode, where the layers compute in float and the scalers measure the ranges."              , run_calibration),
    ]
    
    # 各モードを記録する: Quaver かモデルのファイルへの呼び出しごとに、一番近い記録済みの呼び出し元の下にノードを作る
    result = []
    for key, title, about, run in modes:
        top        : dict            = {"children": []}
        open_frames: dict[int, dict] = {}
        
        def profile(frame, event, arg) -> None:
            '''
            #* Quaver かモデルのファイルへの各呼び出しを、一番近い記録済みの呼び出し元の下のノードとして残す。
            '''
            
            # 関数から戻ったらそのフレームを忘れ、Quaver とモデルのファイルの外の呼び出しは飛ばす
            path = Path(frame.f_code.co_filename)
            if event == "return":
                open_frames.pop(id(frame), None)
                return None
            if event != "call" or "<" in frame.f_code.co_qualname or not (path.parent == GUIDE or PACKAGE in path.parents):
                return None
            
            # 記録済みの一番近い呼び出し元を探す
            caller = frame.f_back
            while caller is not None and id(caller) not in open_frames:
                caller = caller.f_back
            
            # その下にこの呼び出しを、module の名前と目的と一緒に足す
            parent                 = top if caller is None else open_frames[id(caller)]
            owner                  = names.get(id(frame.f_locals.get("self")), "")
            file                   = "guide" if path.parent == GUIDE else path.relative_to(ROOT).as_posix()
            node                   = {"name": frame.f_code.co_qualname, "module": owner, "file": file, "note": notes.get((str(path), frame.f_code.co_qualname), ""), "children": []}
            open_frames[id(frame)] = node
            parent["children"].append(node)
            return None
        
        sys.setprofile(profile)
        run()
        sys.setprofile(None)
        
        # 同じ呼び出しの続き（同じ module で、中の呼び出しも同じ）を、回数付きの 1 つのノードにまとめる
        def merge(nodes: list[dict]) -> list[dict]:
            '''
            #* 中の呼び出しも同じ、同じ呼び出しの続きを、回数付きの 1 つのノードにまとめる。
            '''
            
            merged: list[dict] = []
            for node in nodes:
                node["children"] = merge(node["children"])
                if merged and json.dumps([merged[-1]["name"], merged[-1]["module"], merged[-1]["children"]]) == json.dumps([node["name"], node["module"], node["children"]]):
                    merged[-1]["count"] = merged[-1].get("count", 1) + 1
                else:
                    merged.append(node)
            return merged
        result.append({"key": key, "title": title, "about": about, "calls": merge(top["children"])})
    
    # サイト用の JSON を書く
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes((json.dumps({"modes": result}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    print(f"wrote {OUTPUT.relative_to(ROOT).as_posix()}: " + ", ".join(f"{mode['key']} {json.dumps(mode).count(chr(34) + 'name' + chr(34))} calls" for mode in result))
    
    return None


if __name__ == "__main__":
    flow_trace_main()
