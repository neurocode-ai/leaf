# leaf, the minimal deep learning framework
![Builds status badge](https://github.com/neurocode-ai/leaf/actions/workflows/builds.yml/badge.svg)
![Unit tests status badge](https://github.com/neurocode-ai/leaf/actions/workflows/unittests.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

In its essence, leaf is a [NumPy](https://numpy.org/) only implementation of the
well known [PyTorch](https://pytorch.org/) framework. The convolutional operations
accelerated with Rust binaries enable a robust API for deep learning without GPU. 
Support for GPU is provided through [pyopencl](https://documen.tician.de/pyopencl/)
which allows access to parallel compute devices in Python.

## Setup and install
The first thing you are going to need is a Rust compiler. Easiest way to install it
is by using the toolchain manager `rustup`. Go to
[this](https://www.rust-lang.org/tools/install) link and following the instructions.


The second step is to create and activate a virtual environment. This can be done
with the `venv` module like this.
```
$ python3 -m venv my-venv
$ source my-venv/bin/activate
(my-venv) $ 
```


Next step is to install the required Python libraries. Easiest way to do this is with
`pip` in the following way.
```
$ python3 -m pip install -r requirements.txt
```


Last step is to compile the Rust binary so that we can access the Rust modules from 
our Python code. There is a Makefile for building everything, simply run it by typing
`make` (To remove all compiled binaries and clean up the Rust directory, instead run
`make clean`).


Now you can verify that everything has been set up correctly by running the
`pyvsrust.py` script, which runs a minimal benchmark test on the Rust binaries.

## Example
...
```python
from leaf import Tensor

x = Tensor([[-1.4, 2.3, 5.9]], requires_grad=True)
w = Tensor.eye(3, requires_grad=True)

y = x.dot(w).mean()
y.backward()

print(x.grad)  # dy/dx
print(w.grad)  # dy/dw
```

## Neural networks
...


## License
All code written is to be held under a general MIT license, please see [LICENSE](https://github.com/neurocode-ai/leaf/blob/main/LICENSE) for specific information.
