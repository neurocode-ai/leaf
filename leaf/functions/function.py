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
from leaf import Tensor
from typing import List
from leaf.types import Boolean, String

def _tensors_require_grad(*tensors: List[Tensor]) -> Boolean:
    return any(t.requires_grad for t in tensors if isinstance(t, Tensor))

def _verify_tensors(*tensors: List[Tensor]) -> List[np.ndarray]:
    return [_extract_data(t) for t in tensors]

def _extract_data(tensor: Tensor) -> np.ndarray:
    """ TEMPORARY!!!! EDIT this function """
    return tensor.data

class Function(object):
    def __init__(self, device: String, *tensors: List[Tensor]):
        self.parents = [t for t in tensors if isinstance(t, Tensor)]
        self.device = device
        self.saved_tensors = []
        self.requires_grad = _tensors_require_grad(*tensors)

    def save_for_backward(self, *tensors: List[Tensor]):
        self.saved_tensors.extend(tensors)
    
    def forward(self, *args, **kwargs): 
        raise NotImplementedError(f'forward pass not implemented for {type(self)}')
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError(f'backward pass not implemented for {type(self)}')
    
    @classmethod
    def apply(cls, *tensors: List[Tensor]) -> Tensor: 
        context = cls(*tensors[0].device, *tensors)
        results = Tensor(context.forward(*_verify_tensors(*tensors)),
                         requires_grad=context.requires_grad,
                         device=context.device)
        results._ctx = context
        return results
