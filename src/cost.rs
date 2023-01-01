use std::ops::{Sub, Add};
use ndarray::{Array2, Ix2};
use crate::matrixutil::{power_of, scalar_mult, create_layer};

// enum storing each cost function
pub enum Cost {
    MSE,
}

impl Cost {

    pub fn calculate(&self, predicted: &Vec<Vec<Vec<Array2<f32>>>>, expected: &Vec<Array2<f32>>) -> Array2<f32>{
        match self {
            Cost::MSE => { 
                let mut outp: Array2<f32> = create_layer::<Ix2>(Vec::from(predicted[0][1].last().unwrap().shape()));
                for i in 0..predicted.len() {
                    outp = outp.add(power_of(predicted[i][1].last().unwrap().clone().sub(&expected[i]), 2));
                }
                outp / predicted.len() as f32
            },
        }
    }

    pub fn derivate(&self, predicted: &Vec<Vec<Vec<Array2<f32>>>>, expected: &Vec<Array2<f32>>) -> Array2<f32> {
        match self {
            Cost::MSE => {
                
                let mut outp: Array2<f32> = create_layer::<Ix2>(Vec::from(predicted[0][1].last().unwrap().shape()));
                for i in 0..predicted.len() {
                    println!("pred: {:?}\n\nexpect: {:?}",predicted[i][1].last().unwrap(), expected);
                    outp = outp.add(scalar_mult(predicted[i][1].last().unwrap().clone().sub(&expected[i]), 2f32));
                }
                outp / predicted.len() as f32
            } 
        }
    }
}
