# @file cifar10.py
# @brief Download CIFAR-10, load it onto the GPU, and make random crops and flips there
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
    #* Load CIFAR-10 as uint8 images (N, 3, 32, 32) and int64 labels on the device: train images, train labels, test images, and test labels.
    #? The first time, it downloads the python version of CIFAR-10 (about 170 MB) and unpacks it into folder/cifar-10-batches-py.
    #! The download is checked with the MD5 sum on the CIFAR-10 page. A broken download is deleted, so the next run downloads it again.
    '''
    
    # Download and unpack CIFAR-10 the first time
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
    
    # Read the five train batches and the test batch
    images, labels = [], []
    for name in [f"data_batch_{index}" for index in range(1, 6)] + ["test_batch"]:
        with (batches / name).open("rb") as file:
            batch = pickle.load(file, encoding="bytes")
        images.append(torch.from_numpy(batch[b"data"]).reshape(-1, 3, 32, 32))
        labels.append(torch.tensor(batch[b"labels"]))
    
    # The first five are the train set, and the last one is the test set
    return torch.cat(images[:5]).to(device), torch.cat(labels[:5]).to(device), images[5].to(device), labels[5].to(device)


def cifar10_augment(images: torch.Tensor) -> torch.Tensor:
    '''
    #* Return a random crop (with 4 pixels of zero padding) and a random left-right flip of each image.
    #? train_fit() calls this for each batch. Everything stays on the device.
    '''
    
    # Pad, and take one 32 x 32 window at a random place for each image
    count   = images.shape[0]
    device  = images.device
    padded  = torch.nn.functional.pad(images, (4, 4, 4, 4))
    rows    = torch.randint(0, 9, (count, 1), device=device) + torch.arange(32, device=device)
    columns = torch.randint(0, 9, (count, 1), device=device) + torch.arange(32, device=device)
    crops   = padded[torch.arange(count, device=device)[:, None, None, None], torch.arange(3, device=device)[None, :, None, None], rows[:, None, :, None], columns[:, None, None, :]]
    
    # Flip about half of them
    flipped = torch.rand(count, device=device) < 0.5
    return torch.where(flipped[:, None, None, None], crops.flip(-1), crops)
