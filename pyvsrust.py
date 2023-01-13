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
# Last updated: 2023-01-13
#

import leafrs as rs
import numpy as np
import time
import sys
import argparse
from datetime import datetime

strformat = '%Y-%m-%d %H:%M:%S'
def info(s, strftime=True):
    strf = f'[{datetime.now().strftime(strformat)}] '
    strf = strf + s if strftime else s
    sys.stdout.write(strf)
    sys.stdout.flush()

parser = argparse.ArgumentParser(
    prog='pyvsrust',
    description='Python vs Rust API performance test'
)
parser.add_argument('-d', action='store', nargs=4, required=True)
args = parser.parse_args()

dimensions = tuple([int(d) for d in args.d])

info('Running performance test Python vs Rust API\n')
info(f'Generating test data with shape: {dimensions}...')
data = np.random.rand(*dimensions)
info('OK\n', strftime=False)

pysum = 0
info('Starting Python job...')
t_pystart = time.time()
for h in range(data.shape[2]):
    for w in range(data.shape[3]):
        pysum += data[:, :, h, w]
pysum = pysum.sum()
t_py = time.time() - t_pystart
info('OK\n', strftime=False)
info(f'Time: {t_py:.3f} seconds\n')

info('Starting Rust API job...')
t_rustart = time.time()
rusum = rs.rusum(data).sum()
t_ru = time.time() - t_rustart
info('OK\n', strftime=False)
info(f'Time: {t_ru:.3f} seconds\n')
info(f'Rust took {100.0*(t_ru/t_py):.2f}% of the time Python took\n')
info(f'Rust was {t_py-t_ru:.3f} seconds faster\n')
info(f'Same sum of matrix? {rusum == pysum}\n')
