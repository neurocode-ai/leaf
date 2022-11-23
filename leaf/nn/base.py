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

File created: 2022-11-17
Last updated: 2022-11-23
"""
from leaf import Tensor

class Module(object):
    """ Parent class for the neural network building blocks, i.e. so called Modules.
    They each define specific forward pass functionality based on their needs that 
    is invoked by calling the module object with the input tensor. No __init__ method
    is defined for parent class, please see respective implementations for specific
    information and implementation details.

    """
    def __call__(self, input_) -> Tensor:
        raise self.forward(input_)
    
    def forward(self, input_, *args, **kwargs) -> Tensor:
        """ Each respective neural network module has to implement this depending on funcionality. """
        raise NotImplementedError(
            f'User defined nn.Module {self} has not implemented forward pass.'
        )
    
    def parameters(self) -> list:
        pass

class Sequential(object):
    """ A high-level wrapper for the module object, simplifies the forward pass when
    multiple operations are needed to perform in order. Follows the same naming
    convention as the main module object, namely, __call__() forward() and parameters(),
    that make up the API for neural networks.

    Parameters
    ----------
    *modules: iterable | list | tuple
        The collection of modules stored sequentially.

    """
    def __init__(self, *modules) -> None:
        if not all(isinstance(m, Module) for m in modules):
            raise ValueError(
                f'Not all objects provided to {self} is a module, {modules}.'
            )
        self._modules = modules

    def __call__(self, input_) -> Tensor:
        return self.forward(input_)

    def forward(self, input_) -> Tensor:
        """ Return the resulting tensor after applying all sequential forward passes. """
        x = input_
        for module in self._modules:
            x = module(x)
        return x 

    def parameters(self) -> list:
        """ Return a list of all tensor parameters requiring gradient from stored module objects. """
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params
