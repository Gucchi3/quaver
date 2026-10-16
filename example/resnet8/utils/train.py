# @file train.py
# @brief Train and test a model on the CIFAR-10 tensors, as it is or with a CUDA graph
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
    #* Train the model with AdamW and a cosine learning rate, and test it after each epoch. Returns one record for each epoch.
    #? main.py uses this for FP32 and for each quantized model. With cuda_graph, a few warm-up steps run first, and then one step is captured and replayed for every batch.
    #! The last part of the train set that does not fill a batch is left out, so every step has the same shape (a CUDA graph needs it).
    '''
    
    # Take the data apart
    train_images, train_labels, test_images, test_labels = data
    
    # The learning rate lives in a tensor, so a captured step reads the value the scheduler writes there
    device    = train_images.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(learning_rate, device=device), weight_decay=1e-4, capturable=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=final_rate)
    
    # Fixed tensors for the batch and for the sums of the epoch (a captured step always uses the same memory)
    images = torch.zeros((BATCH_SIZE, 3, 32, 32), dtype=torch.uint8, device=device)
    labels = torch.zeros(BATCH_SIZE             , dtype=torch.int64, device=device)
    totals = torch.zeros(2                      , device=device    )
    steps  = train_images.shape[0] // BATCH_SIZE
    
    # CUDA graph: warm up on a side stream with random batches, and then capture one step on the fixed tensors (the capture itself runs nothing)
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
    
    # Train and test for each epoch
    records = []
    for epoch in range(1, epochs + 1):
        # Copy each shuffled and augmented batch into the fixed tensors, then run the step or replay it
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
        
        # Read the sums (the only GPU sync of the epoch), and time the training part
        loss, correct = totals.tolist()
        seconds       = time.perf_counter() - start
        
        # Test (timed apart from the training), and print one line
        tested      = time.perf_counter()
        accuracy, _ = train_evaluate(model, test_images, test_labels)
        record      = {"epoch": epoch, "loss": loss / (steps * BATCH_SIZE), "train_accuracy": correct / (steps * BATCH_SIZE), "test_accuracy": accuracy, "seconds": seconds, "test_seconds": time.perf_counter() - tested}
        records.append(record)
        print(f"{name:10s} epoch {epoch:3d}/{epochs}  loss {record['loss']:.4f}  train {record['train_accuracy']:.2%}  test {accuracy:.2%}  train time {seconds:.1f} s", flush=True)
    return records


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, images: torch.Tensor, labels: torch.Tensor, totals: torch.Tensor) -> None:
    '''
    #* Run one training step on the batch, and add its loss sum and its number of right answers to totals.
    #? train_fit() runs it as it is, or captures it once in a CUDA graph and replays it.
    #! It makes no GPU sync, so it can be captured. The gradients are set to None first, so a captured backward writes them in place.
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
    #* Return the accuracy and all the logits, in eval mode and without gradients.
    #? main.py compares the logits of the training mode and of the integer mode with this. The integer mode gives codes and their parameters, which params_values() turns into the logits.
    '''
    
    model.eval()
    with torch.no_grad():
        outputs = [model(images[begin:begin + TEST_BATCH]) for begin in range(0, images.shape[0], TEST_BATCH)]
        logits  = torch.cat([params_values(*output) if isinstance(output, tuple) else output for output in outputs])
    return float((logits.argmax(dim=1) == labels).float().mean()), logits
