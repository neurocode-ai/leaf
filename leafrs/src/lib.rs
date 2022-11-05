/*
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
Last edited:  2022-11-05

Tutorial on how to bind Python and Rust via NumPy
https://itnext.io/how-to-bind-python-numpy-with-rust-ndarray-2efa5717ed21
*/

use ndarray;
use numpy::{
    IntoPyArray,
    PyArray1,
    PyArray2,
    PyReadonlyArrayDyn,
    PyReadonlyArray4
};
use pyo3::prelude::{
    pymodule,
    PyModule,
    PyResult,
    Python
};

mod rust_fn {
    use ndarray::{arr1, Array1, Array2};
    use ndarray::prelude::*;
    use numpy::ndarray::{ArrayViewD, ArrayView4};
    use ordered_float::OrderedFloat;

    pub fn max_min(x: &ArrayViewD<'_, f64>) -> Array1<f64> {
        if x.len() == 0 { return arr1(&[]); }
        let max_val = x
            .iter()
            .map(|a| OrderedFloat(*a))
            .max()
            .expect("Error calculating max value.")
            .0;
        let min_val = x
            .iter()
            .map(|a| OrderedFloat(*a))
            .min()
            .expect("Error calculating min value.")
            .0;
        let result_array = arr1(&[max_val, min_val]);
        result_array
    }

    pub fn rusum(x: &ArrayView4<'_, f64>) -> Array2<f64> {
        let xshape = x.shape();
        let mut result_array = Array2::zeros((xshape[0], xshape[1]));
        for h in 0..xshape[2] {
            for w in 0..xshape[3] {
                result_array += &ArrayView::from(x.slice(s![.., .., h, w]));
            }
        }
        result_array
    }
}

#[pymodule]
fn leafrs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn max_min<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>
    ) -> &'py PyArray1<f64> {
        let array = x.as_array();
        let result_array = rust_fn::max_min(&array);
        result_array.into_pyarray(py)
    }

    #[pyfn(m)]
    fn rusum<'py>(
        py: Python<'py>,
        x: PyReadonlyArray4<f64>
    ) -> &'py PyArray2<f64> {
        let array = x.as_array();
        let rustsum = rust_fn::rusum(&array);
        rustsum.into_pyarray(py)
    }

    #[pyfn(m)]
    fn eye<'py>(
        py: Python<'py>,
        size: usize
    ) -> &PyArray2<f64> {
        let array = ndarray::Array::eye(size);
        array.into_pyarray(py)
    }

    Ok(())
}
