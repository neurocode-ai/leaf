import rust_numpy as rn
import numpy as np

data = np.random.rand(100, 100)
print(data.shape)
print(f'max:\t{np.max(data)}')
print(f'max:\t{np.max(data)}')
print(f'maxmin:\t{rn.max_min(data)}') 

