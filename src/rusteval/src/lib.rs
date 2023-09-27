// rustimport:pyo3

mod evaluator;
// mod constant_optimizer;

use pyo3::prelude::*;

use crate::evaluator::Evaluator;
// use crate::constant_optimizer::Optimizer;




/// A Python module implemented in Rust.
#[pymodule]
fn rusteval(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Evaluator>()?;
    // m.add_class::<Optimizer>()?;
    Ok(())
}
