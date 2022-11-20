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
import unittest
from leaf import Tensor

class TestOps(unittest.TestCase):
    def test_add1(self):
        t1 = Tensor.ones(2, 3, 4)
        t2 = Tensor.ones(2, 3, 4)
        t3 = t1.add(t2)

        assert t3.shape == (2, 3, 4)

    def test_sub1(self):
        t1 = Tensor.ones(2, 3, 4)
        t2 = Tensor.ones(2, 3, 4)
        t3 = t1.sub(t2)

        assert t3.shape == (2, 3, 4)
        assert (t3.data == Tensor.zeros(2, 3, 4).data).all()
