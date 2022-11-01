use ndarray;
use numpy::{
    IntoPyArray,
    PyArray1,
    PyArray2,
    PyArray4,
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
    use ndarray::{arr1, Array1, Array2, Array4};
    use numpy::ndarray::{ArrayViewD};
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

    pub fn rusum(x: &Array4<f64>) -> Array2<f64> {
        let rustsum = Array2
        for h in 0..256 {
            for w in 0..256 {

            }
        }
    }
}

#[pymodule]
fn rust_numpy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn max_min<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>
    ) -> &'py PyArray1<f64> {
        let array = x.as_array();
        let result_array = rust_fn::max_min(&array);
        result_array.into_pyarray(py)
    }

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
