# @file cifar10.py
# @brief CIFAR-10 をダウンロードして GPU に読み込み、ランダムな切り抜きと反転を GPU の上で作る
# @author Kohei Yamaguchi, Claude

from pathlib import Path

import hashlib
import pickle
import tarfile
import torch
import urllib.request

from .exceptions import Error

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"



def cifar10_load(folder: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    #* CIFAR-10 を、uint8 の画像 (N, 3, 32, 32) と int64 のラベルとしてデバイスに読み込む。学習用の画像、学習用のラベル、テスト用の画像、テスト用のラベルを返す。
    #? 初回は CIFAR-10 の python 版（約 170 MB）をダウンロードし、folder/cifar-10-batches-py に展開する。
    #! ダウンロードしたファイルは、CIFAR-10 のページにある MD5 で確かめる。壊れていたら消すので、次に動かしたときにダウンロードし直す。
    '''
    
    # 初回は CIFAR-10 をダウンロードして展開する
    batches = folder / "cifar-10-batches-py"
    if not batches.exists():
        archive = folder / "cifar-10-python.tar.gz"
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}", flush=True)
        urllib.request.urlretrieve(CIFAR10_URL, archive)
        if hashlib.md5(archive.read_bytes()).hexdigest() != CIFAR10_MD5:
            archive.unlink()
            raise Error("The download of CIFAR-10 is broken.", detail=f"The MD5 sum of {archive.name} is not {CIFAR10_MD5}.", hint="Run it again to download it again.", code=4)
        with tarfile.open(archive) as file:
            file.extractall(folder, filter="data")
        archive.unlink()
    
    # 学習用の 5 つの batch と、テスト用の batch を読む
    images, labels = [], []
    for name in [f"data_batch_{index}" for index in range(1, 6)] + ["test_batch"]:
        with (batches / name).open("rb") as file:
            batch = pickle.load(file, encoding="bytes")
        images.append(torch.from_numpy(batch[b"data"]).reshape(-1, 3, 32, 32))
        labels.append(torch.tensor(batch[b"labels"]))
    
    # 初めの 5 つが学習用、最後の 1 つがテスト用
    return torch.cat(images[:5]).to(device), torch.cat(labels[:5]).to(device), images[5].to(device), labels[5].to(device)


def cifar10_augment(images: torch.Tensor) -> torch.Tensor:
    '''
    #* 各画像を、ランダムに切り抜き（周りに 4 画素の 0 を足してから）、ランダムに左右反転したものを返す。
    #? train_fit() が batch ごとに呼ぶ。すべてデバイスの上で行う。
    '''
    
    # 周りを埋め、各画像からランダムな場所の 32 x 32 の窓を 1 つ取る
    count   = images.shape[0]
    device  = images.device
    padded  = torch.nn.functional.pad(images, (4, 4, 4, 4))
    rows    = torch.randint(0, 9, (count, 1), device=device) + torch.arange(32, device=device)
    columns = torch.randint(0, 9, (count, 1), device=device) + torch.arange(32, device=device)
    crops   = padded[torch.arange(count, device=device)[:, None, None, None], torch.arange(3, device=device)[None, :, None, None], rows[:, None, :, None], columns[:, None, None, :]]
    
    # 約半分を反転する
    flipped = torch.rand(count, device=device) < 0.5
    return torch.where(flipped[:, None, None, None], crops.flip(-1), crops)
