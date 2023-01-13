#
# MIT License
#
# Copyright (c) 2022 Neurocode
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# File created: 2022-11-05
# Last updated: 2023-01-13
#

import numpy as np
from leaf import Tensor
from typing import List
from leaf.types import Boolean, String

def _tensors_require_grad(*tensors) -> Boolean:
    """ Return True if any tensor requires gradient calculation. """
    return any(t.requires_grad for t in tensors if isinstance(t, Tensor))

def _verify_tensors(*tensors) -> List[np.ndarray]:
    """ Return a list of the data stored in the tensors. """
    return [_extract_data(t) for t in tensors]

def _extract_data(tensor) -> np.ndarray:
    """ TEMPORARY!!!! EDIT this function """
    return tensor.data

class Function(object):
    """ Definition and impelmentation of the Function class.

    Parameters
    ----------
    device: str
        A string representing the device to put resulting tensor on.
    *tensors: iterable | list | tuple
        Collection of tensors that are to be used in the defined operation. 
        For example, two tensors if a binary op is invoked.

    """
    def __init__(self, device, *tensors) -> None:
        self.parents = [t for t in tensors if isinstance(t, Tensor)]
        self.device = device
        self.saved_tensors = []
        self.requires_grad = _tensors_require_grad(*tensors)

    def save_for_backward(self, *items) -> None:
        """ Store the provided items during forward pass to later be used. """
        self.saved_tensors.extend(items)
    
    def forward(self, *args, **kwargs): 
        raise NotImplementedError(f'forward pass not implemented for {type(self)}')
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError(f'backward pass not implemented for {type(self)}')
    
    @classmethod
    def apply(cls, *tensors) -> Tensor: 
        """ This classmethod constructs a Function, also referred to as a context, when 
        a tensor invokes an operation. As such, the tensor initially invoking the 
        call, self, is represented as part of the *tensors arg together with any
        optional tensors that are to be part of the context. 

        After applying the the op on the provided tensors, a resulting tensor is 
        created and returned. This tensor is not a leaf node in the DAG, and 
        contains the context of the op, meaning, it specifies that it was 
        created through an op and stores the parent tensors.

        """
        context = cls(*tensors[0].device, *tensors)
        results = Tensor(context.forward(*_verify_tensors(*tensors)),
                         requires_grad=context.requires_grad,
                         device=context.device)
        results._ctx = context
        return results
