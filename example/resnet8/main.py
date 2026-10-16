# @file main.py
# @brief Train ResNet-8 on CIFAR-10 in FP32, fine-tune the quantized models with QAT, and check their integer mode
# @author Kohei Yamaguchi, Claude

from pathlib import Path

import argparse
import json
import torch

from quaver import fold_batch_norms, freeze_scalers, precision_disable_tf32, precision_set_math, switch_integer_mode
from .model import MODEL_REGISTRY, Resnet8, build_model
from .utils import BATCH_SIZE, Error, cifar10_load, train_evaluate, train_fit

FOLDER = Path(__file__).resolve().parent



def main() -> None:
    '''
    #* Train FP32 ResNet-8 (or load it), fine-tune each quantized model with QAT, check its integer mode, and save the weights and a table of the results.
    #? Run it from the top of the repository: python -m example.resnet8.main (pick models with --models, train faster with --cuda-graph --float32, try quickly with --fp32-epochs 1 --qat-epochs 1).
    #! Needs a GPU. CIFAR-10 is downloaded into data/ the first time, and the weights and the results go to log/.
    '''
    
    # Read the settings: the quantized models to train after FP32 (all of them by default, and none with only --models)
    parser = argparse.ArgumentParser(description="Train ResNet-8 on CIFAR-10 in FP32, fine-tune the quantized models with QAT, and check their integer mode.")
    parser.add_argument("--models", nargs="*", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY), help="quantized models to train after FP32")
    
    # The numbers of epochs
    parser.add_argument("--fp32-epochs" , type=int, default=40, help="epochs of FP32 training"                                 )
    parser.add_argument("--qat-epochs"  , type=int, default=15, help="epochs of QAT for each model"                            )
    parser.add_argument("--freeze-epoch", type=int, default=3 , help="the activation ranges stop after this many epochs of QAT")
    
    # How to run each step: as it is or a replay of one captured step (CUDA graph), and the float type of the training math (the tests stay float64)
    parser.add_argument("--cuda-graph", action="store_true", help="capture one training step in a CUDA graph and replay it"        )
    parser.add_argument("--float32"   , action="store_true", help="train with float32 math (about 1.6x faster); tests stay float64")
    args = parser.parse_args()
    
    # Check the settings here at the entry
    if not torch.cuda.is_available():
        raise Error("This example needs a GPU.", hint="Run it on a machine with CUDA.", code=1)
    if min(args.fp32_epochs, args.qat_epochs) < 1 or args.freeze_epoch < 0:
        raise Error("The numbers of epochs are wrong.", detail=f"--fp32-epochs {args.fp32_epochs}, --qat-epochs {args.qat_epochs}, --freeze-epoch {args.freeze_epoch}", hint="Use 1 or more for the epochs, and 0 or more for --freeze-epoch.", code=2)
    
    # Exact float math for the integer accumulators (float32 only in the training steps if asked), a fixed seed, and the data on the GPU
    precision_disable_tf32()
    precision_set_math(train="float32" if args.float32 else "float64")
    torch.manual_seed(0)
    device = torch.device("cuda")
    data   = cifar10_load(FOLDER / "data", device)
    steps  = data[0].shape[0] // BATCH_SIZE
    log    = FOLDER / "log"
    log.mkdir(exist_ok=True)
    
    # FP32: train it, or load the saved weights (delete the file to train it again)
    fp32 = Resnet8().to(device)
    path = log / "resnet8_fp32.pth"
    if path.exists():
        fp32.load_state_dict(torch.load(path, map_location=device))
        history = []
    else:
        history = train_fit(fp32, data, args.fp32_epochs, 1e-3, 1e-5, "fp32", args.cuda_graph)
        torch.save(fp32.state_dict(), path)
    
    # The results start with FP32
    results: dict[str, dict] = {"fp32": {"weight": "fp32", "activation": "fp32", "accuracy": train_evaluate(fp32, data[2], data[3])[0], "history": history}}
    
    # QAT for each model, from the FP32 weights with the BatchNorms folded in
    folded = fold_batch_norms(fp32).state_dict()
    for name in args.models:
        # Load the folded weights (only the ranges of the quantizers and the input scale are new)
        model               = build_model(name).to(device)
        missing, unexpected = model.load_state_dict(folded, strict=False)
        if unexpected or any(not key.endswith((".kept", ".updates", "input_scale")) for key in missing):
            raise Error("The FP32 weights do not fit the model.", detail=f"{name}: missing {missing}, unexpected {unexpected}", code=3)
        
        # Train (the activation ranges stop by themselves after --freeze-epoch epochs), and save the weights
        freeze_scalers(model, step=args.freeze_epoch * steps)
        history = train_fit(model, data, args.qat_epochs, 1e-4, 1e-6, name, args.cuda_graph)
        torch.save(model.state_dict(), log / f"resnet8_{name}.pth")
        
        # Test in the training mode, and then in the integer mode (the logits must be exactly the same)
        fake_accuracy, fake_logits       = train_evaluate(model, data[2], data[3])
        switch_integer_mode(model, True)
        integer_accuracy, integer_logits = train_evaluate(model, data[2], data[3])
        switch_integer_mode(model, False)
        
        # Keep the results of the model (asym marks the formats with zero points, and blocks the block scales of the block convs)
        setting       = MODEL_REGISTRY[name]
        blocks        = f" + {setting.scale_format} / 16" if setting.block_scaled else ""
        weight        = setting.weight_format + blocks + ("" if setting.weight_symmetric else " asym")
        activation    = setting.act_format + blocks + ("" if setting.act_symmetric else " asym")
        results[name] = {"weight": weight, "activation": activation, "accuracy": fake_accuracy, "integer_accuracy": integer_accuracy, "modes_match": torch.equal(fake_logits, integer_logits), "history": history}
    
    # Show the table (with the mean training time of one epoch), and save it with the full history
    lines = ["| model | weight | activation | test accuracy | integer mode | modes match | train time / epoch |", "|---|---|---|---|---|---|---|"]
    for name, result in results.items():
        integer = f"{result['integer_accuracy']:.2%}" if "integer_accuracy" in result else "-"
        match   = str(result.get("modes_match", "-"))
        seconds = f"{sum(record['seconds'] for record in result['history']) / len(result['history']):.1f} s" if result["history"] else "-"
        lines.append(f"| {name} | {result['weight']} | {result['activation']} | {result['accuracy']:.2%} | {integer} | {match} | {seconds} |")
    print("\n".join(lines))
    (log / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (log / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    
    return None


if __name__ == "__main__":
    main()
