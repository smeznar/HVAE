// rustimport:pyo3

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use ndarray::Array1;
use std::collections::HashMap;


#[pyclass]
struct Evaluator {
    data: HashMap<String, Array1<f64>>,
    true_val: Array1<f64>,
    var_len: usize
}

impl Evaluator {
   fn _eval_expr(&self, expr: Vec<&str>, constants: Vec<f64>) -> PyResult<Array1<f64>>{
       let mut cns = constants.into_iter();
       let mut stack: Vec<Array1<f64>> = Vec::new();
        for t in expr {
            match t {
                "+" => {
                    let val = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))? + stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(val)
                }
                "-" => {
                    let b = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a-b)
                }
                "*" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))? * stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a)
                }
                "/" => {
                    let b = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))? / b;
                    if a.iter().all(|v| v.is_finite()){
                        stack.push(a);
                    } else {
                        return Err(PyValueError::new_err("Exception during evaluation: Division by zero."))
                    }
                }
                "log" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    let b = a.map(|v| v.log10());
                    if b.iter().all(|v| v.is_finite()){
                        stack.push(b);
                    } else {
                        return Err(PyValueError::new_err("Exception during evaluation: Log of a negative number."))
                    }
                }
                "sqrt" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    let b = a.map(|v| v.sqrt());
                    if b.iter().all(|v| v.is_finite()){
                        stack.push(b);
                    } else {
                        return Err(PyValueError::new_err("Exception during evaluation: Square root of a negative number."))
                    }
                }
                "cos" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.cos()));
                }
                "sin" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.sin()));
                }
                "exp" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.exp()));
                }
                "^2" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.powi(2)));
                }
                "^3" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.powi(3)));
                }
                "^4" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.powi(4)));
                }
                "^5" => {
                    let a = stack.pop().ok_or(PyValueError::new_err("Exception during evaluation: Ran out of tokens in the stack."))?;
                    stack.push(a.map(|v| v.powi(5)));
                }
                "C" => {
                    let c = cns.next().ok_or(PyValueError::new_err("Exception during evaluation: Not enough constants were given."))?;
                    stack.push(Array1::<f64>::zeros(self.var_len).map(|x| x+c));
                }
                _ => {
                    if self.data.contains_key(t) {
                        let variable = self.data.get(t).unwrap().clone();
                        stack.push(variable);
                    }
                    else {
                        match t.parse::<f64>() {
                            Ok(v) => stack.push(Array1::<f64>::zeros(self.var_len).map(|x| x+v)),
                            Err(_) => return Err(PyValueError::new_err(format!("Exception during evaluation: Token {} not found, check tokens or the rusteval/src/lib.rs file.", t)))
                        }
                    }
                }
            }
        }
        if stack.len() == 1 {
            return Ok(stack.pop().unwrap());
        } else {
            return Err(PyValueError::new_err("Exception during evaluation: More than one value in the evaluation stack at the end (some tokens might be missing)."))
        }
   }
}

#[pymethods]
impl Evaluator {
    #[new]
    fn new(data: Vec<Vec<f64>>, names: Vec<String>, target: Vec<f64>) -> PyResult<Self> {
        if data.len() != names.len() {
            return Err(PyValueError::new_err("Exception during initialization of the Evaluator: Arguments data and names not of the same length."))
        }
        let var_len = if data.len() > 0 {data.get(0).unwrap().len()} else {1};

        if target.len() != var_len {
            return Err(PyValueError::new_err("Exception during initialization of the Evaluator: Target and data vectors not of the same length."))
        }

        let mut hm = HashMap::new();
        for (d, n) in data.into_iter().zip(names) {
            hm.insert(n, Array1::from_vec(d));
        }

        let true_val = Array1::from_vec(target);

        return Ok(Evaluator { data: hm, var_len, true_val})
    }

    fn eval_expr(&self, expr: Vec<&str>, constants: Vec<f64>) -> PyResult<Vec<f64>>{
        match self._eval_expr(expr, constants) {
            Ok(v) => Ok(v.into_raw_vec()),
            Err(x) => Err(x)
        }
    }

    fn get_rmse(&self, expr: Vec<&str>, constants: Vec<f64>) -> PyResult<f64>{
        match self._eval_expr(expr, constants){
            Err(x) => return Err(x),
            Ok(pv) => {
                let s: f64 = (self.true_val.clone() - pv)
                    .into_iter()
                    .map(|v| f64::powi(v, 2))
                    .sum();
                let rmse = s / (self.var_len as f64);
                if rmse.is_finite() {
                    return Ok(rmse)
                } else {
                    return Err(PyValueError::new_err("Exception during evaluation: Overflow when calculating rmse."))
                }
            }
        }
    }



}


/// A Python module implemented in Rust.
#[pymodule]
fn rusteval(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Evaluator>()?;
    Ok(())
}