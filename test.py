import rust_numpy as rn
import numpy as np
import time

data = np.random.rand(512, 3, 256, 256)
print(data.shape)

pysum = 0
t_pystart = time.time()
for h in range(256):
    for w in range(256):
        pysum += data[:, :, 0, 0]
t_py = time.time() - t_pystart
print(f'Python time = {t_py} , sum = {pysum.sum()}')

t_rustart = time.time()
rusum = rn.rusum(data).sum()
t_ru = time.time() - t_rustart
print(f'Rust time = {t_ru} , sum = {rusum}')
