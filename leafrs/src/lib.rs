//
// MIT License
//
// Copyright (c) 2022 Neurocode
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// File created: 2022-11-01
// Last updated: 2023-01-13
//

use numpy::{
    IntoPyArray,
    PyArray2,
    PyArrayDyn,
    PyReadonlyArrayDyn,
    PyReadonlyArray4
};
use pyo3::prelude::{
    pymodule,
    PyModule,
    PyResult,
    Python
};

// Declare the scope/module for function specific implementations, called from Python
// through the created maturin bindings of the `PyO3` crate. Inside this scope
// we operate on the dynamic arrays specified by the `ndarray` crate. The `numpy`
// crate used in the outer scope is only used as API for the native C bindings.
mod RUST_BACKEND {
    use ndarray::{Array2, ArrayD};
    use ndarray::prelude::*;
    use numpy::ndarray::{ArrayViewD, ArrayView4};

    // BACKEND FUNC, only used in Python vs Rust performance example.
    // Calculates the sum of a 4 dimensional f32 matrix.
    pub fn _example_matrix_sum(x: &ArrayView4<'_, f32>) -> Array2<f32> {
        let xshape = x.shape();
        let mut result_array = Array2::zeros((xshape[0], xshape[1]));
        for h in 0..xshape[2] {
            for w in 0..xshape[3] {
                result_array += &ArrayView::from(x.slice(s![.., .., h, w]));
            }
        }
        result_array
    }

    // BACKEND FUNC, performs addition on two ndarrays.
    pub fn add(
        x: &ArrayViewD<'_, f32>,
        y: &ArrayViewD<'_, f32>
    ) -> ArrayD<f32> {
        x + y
    }

    // BACKEND FUNC, performs subtraction on two ndarrays.
    // z = y + x
    pub fn sub(
        x: &ArrayViewD<'_, f32>,
        y: &ArrayViewD<'_, f32>
    ) -> ArrayD<f32> {
        x - y
    }
}

#[pymodule]
fn leafrs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // Unary operator, calculates the sum of a 4D matrix/tensor.
    // ONLY USED IN EXAMPLE SCRIPT COMPARING PYTHON AND RUST SPEED!
    #[pyfn(m)]
    fn example_matrix_sum<'py>(
        py: Python<'py>,
        x: PyReadonlyArray4<f32>
    ) -> &'py PyArray2<f32> {
        let arr = x.as_array();
        let sum = RUST_BACKEND::_example_matrix_sum(&arr);
        sum.into_pyarray(py)
    }

    // Binary operator, performs addition on two ndarrays.
    #[pyfn(m)]
    fn add<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f32>,
        y: PyReadonlyArrayDyn<f32>
    ) -> &'py PyArrayDyn<f32> {
        let result = RUST_BACKEND::add(&x.as_array(), &y.as_array());
        result.into_pyarray(py)
    }

    // Binary operator, performs subtraction on two ndarrays.
    #[pyfn(m)]
    fn sub<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f32>,
        y: PyReadonlyArrayDyn<f32>
    ) -> &'py PyArrayDyn<f32> {
        let result = RUST_BACKEND::sub(&x.as_array(), &y.as_array());
        result.into_pyarray(py)
    }

    Ok(())
}
