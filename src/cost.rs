use ndarray::{Array2};
use crate::matrixutil::{power_of, scalar_mult};

// enum storing each cost function
pub enum Cost {
    SE,
}

impl Cost {
    pub fn calculate(&self, predicted: Array2<f32>, expected: Array2<f32>) -> Array2<f32>{
        match self {
            Cost::SE => { power_of(predicted - expected, 2) },
        }
    }

    pub fn derivate(&self, predicted: Array2<f32>, expected: Array2<f32>) -> Array2<f32> {
        match self {
            Cost::SE => { scalar_mult(predicted - expected, 2f32) },
        } 
    }
}
