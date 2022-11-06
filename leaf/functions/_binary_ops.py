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
Last edited:  2022-11-05
"""
import numpy as np
import leafrs as rs
from .function import Function
from typing import Tuple

def _unbroadcast(arr: np.ndarray, shape: Tuple) -> np.ndarray:
    """ Revert broadcasting on an array to desired original shape. """
    return np.lib.stride_tricks.as_stride(arr, shape).copy()

class Add(Function):
    def forward(self, x, y) -> np.ndarray:
        self.save_for_backward(x.shape, y.shape)
        return rs.add(x, y)
    
    def backward(self, grad) -> Tuple[np.ndarray]:
        xshape, yshape, = self.saved_tensors
        return _unbroadcast(grad, xshape), _unbroadcast(grad, yshape)
