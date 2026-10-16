# @file flow_trace.py
# @brief Record which Quaver functions the model of the guides calls, in order, in each mode, for the flow page of the API reference
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
    #* Run the model in each mode under a profiler, keep the calls into Quaver as a tree, and write it as JSON for the site.
    #? The tree keeps the calls of the functions of quaver/ and of the model files next to this script. The #* line of each docstring says what the call is for.
    #! Run it from the top of the repository after the code changes: uv run python website/scripts/flow_trace.py
    '''
    
    # The #* line of every function and method, by file and qualified name
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
    
    # Build the model, set its scales with a batch, and name its modules (conv1, depthwise2, ...)
    precision_disable_tf32()
    torch.manual_seed(0)
    model  = TinyNet()
    images = torch.randn(2, 3, 32, 32)
    calibrate_model(model, [images])
    model(images)
    names  = {id(module): name or "model" for name, module in model.named_modules()}
    
    # The modes to record: a key, a title, what it shows, and how to run it
    def run_training() -> None:
        '''
        #* One QAT step: forward and backward in train().
        '''
        
        model.train()
        model(images).sum().backward()
        return None
    
    def run_test() -> None:
        '''
        #* A test: forward in eval() without gradients.
        '''
        
        model.eval()
        with torch.no_grad():
            model(images)
        return None
    
    def run_integer() -> None:
        '''
        #* The integer mode: forward on codes, then back to the training mode.
        '''
        
        model.eval()
        switch_integer_mode(model, True)
        with torch.no_grad():
            model(images)
        switch_integer_mode(model, False)
        return None
    
    def run_calibration() -> None:
        '''
        #* Calibration: calibrate_model() in the float mode.
        '''
        
        calibrate_model(model, [images])
        return None
    
    modes = [
        ("training"   , "Training step", "One forward pass of QAT, in train(): the scalers update, and every step that rounds runs the integer math of the device.", run_training   ),
        ("test"       , "Test"         , "The same model in eval() and torch.no_grad(): the scales stay fixed."                                                  , run_test       ),
        ("integer"    , "Integer mode" , "The model on integer codes, with integer math only, as on the device."                                                 , run_integer    ),
        ("calibration", "Calibration"  , "calibrate_model(): the float mode, where the layers compute in float and the scalers measure the ranges."              , run_calibration),
    ]
    
    # Record each mode: a node for every call into Quaver or the model files, under the nearest recorded caller
    result = []
    for key, title, about, run in modes:
        top        : dict            = {"children": []}
        open_frames: dict[int, dict] = {}
        
        def profile(frame, event, arg) -> None:
            '''
            #* Keep each call into Quaver or the model files as a node under its nearest recorded caller.
            '''
            
            # Forget a frame when it returns, and skip calls outside Quaver and the model files
            path = Path(frame.f_code.co_filename)
            if event == "return":
                open_frames.pop(id(frame), None)
                return None
            if event != "call" or "<" in frame.f_code.co_qualname or not (path.parent == GUIDE or PACKAGE in path.parents):
                return None
            
            # Find the nearest caller that is recorded
            caller = frame.f_back
            while caller is not None and id(caller) not in open_frames:
                caller = caller.f_back
            
            # Add this call under it, with the name of its module and what it is for
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
        
        # Merge runs of the same call (same module, same calls inside) into one node with a count
        def merge(nodes: list[dict]) -> list[dict]:
            '''
            #* Merge runs of the same call with the same calls inside into one node with a count.
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
    
    # Write the JSON for the site
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes((json.dumps({"modes": result}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    print(f"wrote {OUTPUT.relative_to(ROOT).as_posix()}: " + ", ".join(f"{mode['key']} {json.dumps(mode).count(chr(34) + 'name' + chr(34))} calls" for mode in result))
    
    return None


if __name__ == "__main__":
    flow_trace_main()
