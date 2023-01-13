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
# File created: 2022-11-18
# Last updated: 2023-01-13
#

from leaf import Tensor
from leaf.nn import Module

class Linear(Module):
    """ Linear layer implementation as a neural network module.
    This module requires input to be 2D tensor, allowing standard matmul op.

    Parameters
    ----------
    fan_in: int
        Dimensionality of the input, i.e., the number of input features.
    fan_out: int
        Wanted dimensionality of the output, latent space dim, second axis.
    bias: bool
        Specify whether or not to use bias parameter in the linear layer,
        if `True`, then learnable bias parameter is added to output of
        the matmul operator.

    """
    def __init__(self, fan_in, fan_out, bias=True) -> None:
        self._weights = Tensor.uniform(fan_in, fan_out, requires_grad=True)
        self._bias = Tensor.uniform(fan_out, requires_grad=True) if bias else None
    
    def forward(self, input_) -> Tensor:
        """ Propagate data through a linear layer, performing linear transform operation.

        Parameters
        ----------
        Input   x: (batch_size, fan_in)
        Weight  W: (fan_in, fan_out)
        Bias    b: (fan_out, )
        Output  y: (batch_size, fan_out)

        y = x @ w + b
        (batch_size, fan_out) = (batch_size, fan_in) @ (fan_in, fan_out) + (fan_out, )

        """
        x = input_.matmul(self._weights)

        if self._bias is None:
            return x
        
        return x.add(self._bias)
