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

File created: 2022-11-01
Last updated: 2022-11-18
"""

from leaf.tensor import Tensor
from leaf import nn
from leaf.types import *

def is_tensor(obj) -> Boolean:
    """ Returns True if `obj` is a tensor.

    Parameters
    ----------
    obj: object
        The object to test.

    """
    return isinstance(obj, Tensor)

def are_tensors(objs) -> Boolean:
    """ Returns True if all objects in `objs` are tensors.

    Parameters
    ----------
    objs: list | tuple | iterable
        The iterable of objects to test.

    """
    return all(map(is_tensor, objs))

# Register all standard CPU ops for tensor.
import os
import inspect
import importlib
from functools import partialmethod

def _register_from_import(namespace) -> None:
    for name, func in inspect.getmembers(namespace, inspect.isclass):
        setattr(Tensor, name.lower(), partialmethod(func.apply))

opfiles = os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'functions'))
opfiles = [f.split('.')[0] for f in opfiles if f.endswith('_ops.py')]
for optype in opfiles:
    try:
        _register_from_import(importlib.import_module('leaf.functions.' + optype))
    except ImportError as error:
        print(f'Could not import module {optype}, {error=}')
