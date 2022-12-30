use ndarray::{Array2};
use crate::matrixutil::{power_of, scalar_mult};

// enum storing each cost function
pub enum Cost {
    MSE,
}

impl Cost {

    pub fn calculate(&self, predicted: Vec<Array2<f32>>, expected: Vec<Array2<f32>>) -> Vec<f32>{
        match self {
            Cost::MSE => { 
                let mut outp: Vec<f32> = Vec::new();
                for i in 0..predicted.len() {
                    outp.push(power_of(predicted[i] - expected[i], 2).mean().unwrap());
                }
                outp
            },
        }
    }

    pub fn derivate(&self, predicted: Array2<f32>, expected: Array2<f32>) -> Array2<f32> {
        match self {
            Cost::MSE => { scalar_mult(predicted - expected, 2f32) },
        } 
    }
}
