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
Last edited: 2022-11-05
"""
import numpy as np
import unittest
from leaf import Tensor

class TestTensor(unittest.TestCase):
    def test_shape(self):
        t1 = Tensor([3, 2, 4])
        t2 = Tensor(2.0)
        t3 = Tensor(np.zeros((3, 4, 2)))
        t4 = Tensor(100)
        t5 = Tensor([[-2.3, 0], [-4.9, 8]])

        assert t1.shape == (3, )
        assert t2.shape == (1, )
        assert t3.shape == (3, 4, 2)
        assert t4.shape == (1, )
        assert t5.shape == (2, 2)

    def test_dtype(self):
        t1 = Tensor([3, 2, 4], dtype=np.int16)
        t2 = Tensor([3, 2.1, 8.6, -2])

        assert t1.dtype == np.int16
        assert t2.dtype == np.float32

