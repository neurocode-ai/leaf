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
# File created: 2022-11-01
# Last updated: 2023-01-14
#

from __future__ import annotations
from typing import Union, Tuple, List
import numpy as np

class Tensor(object):
    """
    Definition and implementation of the Tensor class.

    Parameters
    ----------
    data: list | tuple | int | float | np.ndarray
        The data to create a new Tensor of. Will always be 
        cast to a numpy array after initialization.
    dtype: int | float | np.float32 | np.int16
        The datatype specification for the Tensor. np.float32
        is currently the default datatype, subject to change.
    requires_grad: bool
        Specify whether the Tensor should store gradients
        related to DAG during backwards pass.
    device: str
        Determines on what device the Tensor operations will be
        performed on. Defaults to 'CPU', valid options are
        ('cpu', 'gpu', 'Rust'). GPU currently not supported. 

    """
    def __init__(self, data, *args, dtype=np.float32,
        requires_grad=False, device='cpu', _is_leaf=True, **kwargs) -> None:

        if isinstance(data, (list, tuple)):
            data = np.array(data).astype(dtype)

        if not hasattr(data, '__iter__'):
            data = np.array([data]).astype(dtype)

        if isinstance(data, np.ndarray):
            data = data.astype(dtype)

        self.data = data
        self.grad = None
        self._ctx = None
        self._is_leaf = _is_leaf
        self.device = device
        self.requires_grad = requires_grad

    @property
    def shape(self) -> Tuple:
        return self.data.shape
    
    @property
    def dtype(self) -> Union[np.float32, np.int16]:
        return self.data.dtype

    @classmethod
    def zeros(cls, *shape, **kwargs) -> Tensor:
        return cls(np.zeros(shape), **kwargs)
    
    @classmethod
    def ones(cls, *shape, **kwargs) -> Tensor:
        return cls(np.zeros(shape), **kwargs)
    
    @classmethod
    def eye(cls, dims, **kwargs) -> Tensor:
        return cls(np.eye(dims), **kwargs)
    
    @classmethod
    def uniform(cls, *shape, low=-1.0, high=1.0, **kwargs) -> Tensor:
        return cls(
            np.random.uniform(low, high, size=shape) / np.sqrt(np.prod(shape)),
            **kwargs,
        )

    @classmethod
    def normal(cls, *shape, loc=0.0, scale=1.0, **kwargs):
        return cls(
            np.random.normal(loc, scale, size=shape),
            **kwargs,
        )

    def detach(self) -> Tensor:
        """
        Create a copy of the current tensor that is not part of the 
        dynamic DAG. As such, the new tensor does not, and can not,
        require grad because it is not part of any context nor DAG.
        Subsequentially move the tensor to cpu device, if it was on other.

        """
        return Tensor(self.data, dtype=self.dtype, requires_grad=False, device='cpu')

    def _topological_sort(self):
        def _recursive_walk(node, visited, nodes):
            visited.add(node)
            if node._ctx:
                map(
                    _recursive_walk,
                    zip(
                        [n for n in node._ctx.parents if n not in visited],
                        visited,
                        nodes,
                    )
                )
                nodes.append(node)
            return nodes
        return _recursive_walk(self, set(), [])

    def backward(self, allow_fill=True) -> None:
        """ Calculate gradient backwards through DAG from reduced tensor. """
        if allow_fill:
            self.grad = None
        
        if self._ctx is None:
            return
        
        if self.grad is None and allow_fill:
            assert np.prod(self.shape) == 1, \
                'You are trying to initiate a backwards propagating call on a tensor ' \
                'that has not yet been reduced. The expected behaviour is to reduce ' \
                'your tensor using either ``sum()`` or ``mean``, otherwise implicit ' \
                'creation of the gradient, i.e. initiating it with ones, might be incorrect.'
        
            self.grad = np.ones(self.shape)

        for t in reversed(self._topological_sort()):
            if not any(tt.requires_grad for tt in t._ctx.parents):
                continue
            
            assert t.grad is not None, \
                'The parents of the current node requires gradient, but no gradient has ' \
                'been calculated or initialized for the current node. Make sure that the ' \
                'current node inherited the correct grad requirements from parents.'
            
            parents = t._ctx.parents
            gradients = t._ctx.backward(t.grad)

            if isinstance(gradients, np.ndarray):
                gradients = [gradients]
            
            for gradient, parent in zip(gradients, parents):
                if gradient is None:
                    continue
                    
                if parent.requires_grad:
                    if parent.grad is None:
                        gradient.flags.writeable = True
                        parent.grad = gradient
                    else:
                        if not parent._is_leaf:
                            parent.grad = gradient
                        else:
                            parent.grad += gradient
