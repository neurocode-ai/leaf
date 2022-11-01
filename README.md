# leaf, the minimal deep learning framework
In its essence, leaf is a NumPy only implementation of the well known PyTorch framework. 
The convolutional operations accelerated with Rust binaries enable a robust API for deep learning without GPU.
Support for GPU is provided through pyopencl which allows access to parallel compute devices in Python.

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

## Installation
...

## License
All code written is to be held under a general MIT license, please see [LICENSE](https://github.com/neurocode-ai/leaf/blob/main/LICENSE) for specific information.
