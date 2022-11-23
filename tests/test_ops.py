"""
MIT License

Copyright (c) 2022 Neurocode

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2022-11-05
Last edited:  2022-11-23
"""

import unittest
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timeit
import time
import leaf
from datetime import datetime
from leaf import Tensor
from functools import partial

np.random.seed(1)
torch.manual_seed(1)

strformat = '%Y-%m-%d %H:%M:%S'
def _info(s):
    sys.stdout.write(f'[{datetime.now().strftime(strformat)}] {s}')
    sys.stdout.flush()

def _test_op(shapes, torch_func, leaf_func, name, timeits=10):
    _info(f'testing {name} with shapes {shapes}, torch/leaf \n')
    torch_t = [torch.tensor(np.random.random(size=shape), requires_grad=True) for shape in shapes]
    leaf_t = [Tensor(t.detach().numpy(), requires_grad=True) for t in torch_t]

    torch_out = torch_func(*torch_t)
    leaf_out = leaf_func(*leaf_t)

    _listout_t = isinstance(torch_out, (tuple, list))
    _listout_l = isinstance(leaf_out, (tuple, list))

    if isinstance(torch_out, torch.Tensor):
        torch_out = [torch_out]
    if isinstance(leaf_out, Tensor):
        leaf_out = [leaf_out]
    
    for tt, lt in zip(torch_out, leaf_out):
        np.testing.assert_allclose(
            tt.detach().numpy(),
            lt.data,
            atol=1e-6,
            rtol=1e-3,
        )
        tt.mean().backward()
        lt.mean().backward()

    for tt, lt in zip(torch_t, leaf_t):
        np.testing.assert_allclose(
            tt.detach().numpy(),
            lt.data,
            atol=1e-6,
            rtol=1e-3,
        )
    
    f_torch_ms = timeit.Timer(partial(
        torch_func,
        *torch_t,
    )).timeit(timeits) * 1000.0 / timeits
    f_leaf_ms = timeit.Timer(partial(
        leaf_func,
        *leaf_t
    )).timeit(timeits) * 1000.0 / timeits
    
    b_torch_ms = timeit.Timer(partial(
        lambda f, t, b: f(*t)[0].mean().backward() if b else f(*t).mean().backward(),
        torch_fun,
        torch_t,
        _listout_t,
    )).timeit(timeits) * 1000.0 / timeits
    b_leaf_ms = timeit.Timer(partial(
        lambda f, t, b: f(*t)[0].mean().backward() if b else f(*t).mean().backward(),
        leaf_func,
        leaf_f,
        _listout_l
    )).timeit(timeits) * 1000.0 / timeits

    _info(
        f'forward: {f_torch_ms:.3f} ms / {f_leaf_ms:.3f} ms ' \
        f'backward: {b_torch_ms:.3f} / {b_leaf_ms:.3f} ms\n'
    )

class TestOps(unittest.TestCase):
    def test_add1(self):
        _test_op([(100, 100), (100, 100)], lambda x, y: x + y, Tensor.add, 'add')

    def test_sub1(self):
        _test_op([(100, 100), (100, 100)], lambda x, y: x - y, Tensor.sub, 'sub')
