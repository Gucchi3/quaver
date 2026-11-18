# @file group.py
# @brief テンソルを、1 つの scale を共有するグループに分ける
# @author Kohei Yamaguchi, Claude

import math
import torch

from .types import Granularity



def group_count(shape: torch.Size, granularity: Granularity) -> int:
    '''
    #* この形のテンソルが持つ scale の数を返す。
    '''
    
    if granularity.kind == "tensor":
        return 1
    if granularity.kind == "channel":
        return shape[granularity.axis]
    return math.prod(shape) // granularity.block_size


def group_split(x: torch.Tensor, granularity: Granularity) -> torch.Tensor:
    '''
    #* x を (groups, values) の形にし、各行が 1 つの scale を共有するようにする。
    #? テンソル、チャネル、ブロックの粒度は、すべてこの 1 つの関数を使う。
    #! ブロックでは、軸の大きさは block_size の倍数でなければならない。
    '''
    
    if granularity.kind == "tensor":
        return x.reshape(1, -1)
    if granularity.kind == "channel":
        return x.movedim(granularity.axis, 0).reshape(x.shape[granularity.axis], -1)
    return x.movedim(granularity.axis, -1).reshape(-1, granularity.block_size)


def group_merge(values: torch.Tensor, shape: torch.Size, granularity: Granularity) -> torch.Tensor:
    '''
    #* グループごとに 1 つの値を、shape に broadcast できるテンソルに広げ直す。
    #? テンソルは 0 次元の値を、チャネルは小さな形を返す。軸全体を 1 つのブロックにすると、トークンごとに値を 1 つ返す。
    #! それより小さなブロックは、元と同じ形を返す。
    '''
    
    # 全部で値 1 つ（0 次元なので、x をどう reshape しても合う）か、チャネルごとに 1 つ
    if granularity.kind == "tensor":
        return values.reshape(())
    if granularity.kind == "channel":
        view                   = [1] * len(shape)
        view[granularity.axis] = shape[granularity.axis]
        return values.reshape(view)
    
    # ブロック: ブロックごとに値 1 つ。ブロックが軸全体でなければ、ブロックの中に広げる
    axis   = granularity.axis % len(shape)
    others = [size for index, size in enumerate(shape) if index != axis]
    blocks = shape[axis] // granularity.block_size
    spread = values.reshape(*others, blocks, 1)
    if blocks > 1:
        spread = spread.expand(*others, blocks, granularity.block_size)
    return spread.reshape(*others, -1).movedim(-1, axis)
