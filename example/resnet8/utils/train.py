# @file train.py
# @brief CIFAR-10 のテンソルでモデルを学習しテストする（そのままか、CUDA graph を使う）
# @author Kohei Yamaguchi, Claude

from torch    import nn
from torch.nn import functional as F

import time
import torch

from quaver   import params_values
from .cifar10 import cifar10_augment

BATCH_SIZE   = 128
TEST_BATCH   = 1000
WARMUP_STEPS = 3



def train_fit(model: nn.Module, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], epochs: int, learning_rate: float, final_rate: float, name: str, cuda_graph: bool) -> list[dict[str, float]]:
    '''
    #* AdamW と cosine の学習率でモデルを学習し、エポックごとにテストする。エポックごとに記録を 1 つ返す。
    #? main.py が、FP32 と量子化した各モデルに使う。cuda_graph のときは、先に数 step の準備運転をしてから 1 step を記録し、どの batch でもそれを再生する。
    #! 学習用のデータの最後の、batch に満たない部分は使わないので、どの step も同じ形になる（CUDA graph にはこれが要る）。
    '''
    
    # データを分ける
    train_images, train_labels, test_images, test_labels = data
    
    # 学習率はテンソルに入れておく。記録した step は、scheduler がそこに書いた値を読む
    device    = train_images.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(learning_rate, device=device), weight_decay=1e-4, capturable=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=final_rate)
    
    # batch とエポックの和のための、場所の決まったテンソル（記録した step はいつも同じメモリを使う）
    images = torch.zeros((BATCH_SIZE, 3, 32, 32), dtype=torch.uint8, device=device)
    labels = torch.zeros(BATCH_SIZE             , dtype=torch.int64, device=device)
    totals = torch.zeros(2                      , device=device    )
    steps  = train_images.shape[0] // BATCH_SIZE
    
    # CUDA graph: 別の stream でランダムな batch を使って準備運転をし、場所の決まったテンソルで 1 step を記録する（記録のときは何も計算しない）
    graph = None
    if cuda_graph:
        side  = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(WARMUP_STEPS):
                index = torch.randint(0, train_images.shape[0], (BATCH_SIZE,), device=device)
                images.copy_(cifar10_augment(train_images[index]))
                labels.copy_(train_labels[index])
                train_step(model, optimizer, images, labels, totals)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            train_step(model, optimizer, images, labels, totals)
    
    # エポックごとに学習とテストをする
    records = []
    for epoch in range(1, epochs + 1):
        # シャッフルして水増しした batch を場所の決まったテンソルに写し、step を動かすか再生する
        start = time.perf_counter()
        order = torch.randperm(train_images.shape[0], device=device)
        totals.zero_()
        model.train()
        for step in range(steps):
            index = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            images.copy_(cifar10_augment(train_images[index]))
            labels.copy_(train_labels[index])
            if graph is None:
                train_step(model, optimizer, images, labels, totals)
            else:
                graph.replay()
        scheduler.step()
        
        # 和を読み（エポックの中で GPU の同期はここだけ）、学習の部分の時間を測る
        loss, correct = totals.tolist()
        seconds       = time.perf_counter() - start
        
        # テストし（時間は学習と分けて測る）、1 行表示する
        tested      = time.perf_counter()
        accuracy, _ = train_evaluate(model, test_images, test_labels)
        record      = {"epoch": epoch, "loss": loss / (steps * BATCH_SIZE), "train_accuracy": correct / (steps * BATCH_SIZE), "test_accuracy": accuracy, "seconds": seconds, "test_seconds": time.perf_counter() - tested}
        records.append(record)
        print(f"{name:10s} epoch {epoch:3d}/{epochs}  loss {record['loss']:.4f}  train {record['train_accuracy']:.2%}  test {accuracy:.2%}  train time {seconds:.1f} s", flush=True)
    return records


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, images: torch.Tensor, labels: torch.Tensor, totals: torch.Tensor) -> None:
    '''
    #* batch で学習の step を 1 回動かし、loss の和と正解の数を totals に足す。
    #? train_fit() が、そのまま動かすか、CUDA graph に 1 回記録して再生する。
    #! GPU の同期をしないので、記録できる。勾配は先に None にするので、記録した backward はその場所に勾配を書く。
    '''
    
    logits = model(images)
    loss   = F.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    totals += torch.stack([loss.detach() * labels.shape[0], (logits.argmax(dim=1) == labels).sum().float()])
    
    return None


def train_evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> tuple[float, torch.Tensor]:
    '''
    #* eval mode で勾配なしに、正解率とすべての logits を返す。
    #? main.py がこれで training mode と integer mode の logits を比べる。integer mode は code とそのパラメータを返すので、params_values() で logits に変える。
    '''
    
    model.eval()
    with torch.no_grad():
        outputs = [model(images[begin:begin + TEST_BATCH]) for begin in range(0, images.shape[0], TEST_BATCH)]
        logits  = torch.cat([params_values(*output) if isinstance(output, tuple) else output for output in outputs])
    return float((logits.argmax(dim=1) == labels).float().mean()), logits
