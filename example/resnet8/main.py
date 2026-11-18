# @file main.py
# @brief ResNet-8 を CIFAR-10 で FP32 で学習し、量子化したモデルを QAT で fine-tune して、integer mode を確かめる
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
    #* FP32 の ResNet-8 を学習し（または読み込み）、量子化したモデルをそれぞれ QAT で fine-tune して integer mode を確かめ、重みと結果の表を保存する。
    #? リポジトリの一番上から動かす: python -m example.resnet8.main（モデルは --models で選ぶ。--cuda-graph --float32 で学習が速くなる。--fp32-epochs 1 --qat-epochs 1 で手早く試せる）。
    #! GPU が要る。CIFAR-10 は初回に data/ にダウンロードし、重みと結果は log/ に置く。
    '''
    
    # 設定を読む: FP32 のあとに学習する量子化したモデル（標準ではすべて。--models だけを付けると 0 個）
    parser = argparse.ArgumentParser(description="Train ResNet-8 on CIFAR-10 in FP32, fine-tune the quantized models with QAT, and check their integer mode.")
    parser.add_argument("--models", nargs="*", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY), help="quantized models to train after FP32")
    
    # エポックの数
    parser.add_argument("--fp32-epochs" , type=int, default=40, help="epochs of FP32 training"                                 )
    parser.add_argument("--qat-epochs"  , type=int, default=15, help="epochs of QAT for each model"                            )
    parser.add_argument("--freeze-epoch", type=int, default=3 , help="the activation ranges stop after this many epochs of QAT")
    
    # 各 step の動かし方（そのまま動かすか、記録した 1 step を再生するか（CUDA graph））と、学習の計算の float の型（テストは float64 のまま）
    parser.add_argument("--cuda-graph", action="store_true", help="capture one training step in a CUDA graph and replay it"        )
    parser.add_argument("--float32"   , action="store_true", help="train with float32 math (about 1.6x faster); tests stay float64")
    args = parser.parse_args()
    
    # 設定はこの入口で確かめる
    if not torch.cuda.is_available():
        raise Error("This example needs a GPU.", hint="Run it on a machine with CUDA.", code=1)
    if min(args.fp32_epochs, args.qat_epochs) < 1 or args.freeze_epoch < 0:
        raise Error("The numbers of epochs are wrong.", detail=f"--fp32-epochs {args.fp32_epochs}, --qat-epochs {args.qat_epochs}, --freeze-epoch {args.freeze_epoch}", hint="Use 1 or more for the epochs, and 0 or more for --freeze-epoch.", code=2)
    
    # integer の累積のために float の計算を正確にし（頼まれたときだけ学習の step を float32 にする）、seed を固定して、データを GPU に置く
    precision_disable_tf32()
    precision_set_math(train="float32" if args.float32 else "float64")
    torch.manual_seed(0)
    device = torch.device("cuda")
    data   = cifar10_load(FOLDER / "data", device)
    steps  = data[0].shape[0] // BATCH_SIZE
    log    = FOLDER / "log"
    log.mkdir(exist_ok=True)
    
    # FP32: 学習するか、保存した重みを読み込む（もう一度学習するときはファイルを消す）
    fp32 = Resnet8().to(device)
    path = log / "resnet8_fp32.pth"
    if path.exists():
        fp32.load_state_dict(torch.load(path, map_location=device))
        history = []
    else:
        history = train_fit(fp32, data, args.fp32_epochs, 1e-3, 1e-5, "fp32", args.cuda_graph)
        torch.save(fp32.state_dict(), path)
    
    # 結果の最初は FP32
    results: dict[str, dict] = {"fp32": {"weight": "fp32", "activation": "fp32", "accuracy": train_evaluate(fp32, data[2], data[3])[0], "history": history}}
    
    # モデルごとに、BatchNorm を畳み込んだ FP32 の重みから QAT をする
    folded = fold_batch_norms(fp32).state_dict()
    for name in args.models:
        # 畳み込んだ重みを読み込む（新しいのは、量子化器の範囲と入力の scale だけ）
        model               = build_model(name).to(device)
        missing, unexpected = model.load_state_dict(folded, strict=False)
        if unexpected or any(not key.endswith((".kept", ".updates", "input_scale")) for key in missing):
            raise Error("The FP32 weights do not fit the model.", detail=f"{name}: missing {missing}, unexpected {unexpected}", code=3)
        
        # 学習し（活性化の範囲は --freeze-epoch 回のエポックのあとで自動で止まる）、重みを保存する
        freeze_scalers(model, step=args.freeze_epoch * steps)
        history = train_fit(model, data, args.qat_epochs, 1e-4, 1e-6, name, args.cuda_graph)
        torch.save(model.state_dict(), log / f"resnet8_{name}.pth")
        
        # training mode でテストし、次に integer mode でテストする（logits は完全に同じでなければならない）
        fake_accuracy, fake_logits       = train_evaluate(model, data[2], data[3])
        switch_integer_mode(model, True)
        integer_accuracy, integer_logits = train_evaluate(model, data[2], data[3])
        switch_integer_mode(model, False)
        
        # モデルの結果を残す（asym は zero point のある形式の印、blocks は block conv のブロックの scale の印）
        setting       = MODEL_REGISTRY[name]
        blocks        = f" + {setting.scale_format} / 16" if setting.block_scaled else ""
        weight        = setting.weight_format + blocks + ("" if setting.weight_symmetric else " asym")
        activation    = setting.act_format + blocks + ("" if setting.act_symmetric else " asym")
        results[name] = {"weight": weight, "activation": activation, "accuracy": fake_accuracy, "integer_accuracy": integer_accuracy, "modes_match": torch.equal(fake_logits, integer_logits), "history": history}
    
    # 表を表示し（1 エポックの学習時間の平均も付ける）、全部の記録と一緒に保存する
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
