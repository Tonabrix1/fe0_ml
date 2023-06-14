#![allow(dead_code)]
use std::ops::AddAssign;
use ndarray::Array2;
use crate::{matrixutil::{power_of, scalar_div, create_weight, log_weight}, typings::ForwardBatch};

// enum storing each cost function
pub enum Cost {
    MSE,
    CategoricalCrossEntropy,
}


impl Cost {
    pub fn calculate(&self, predicted: &Array2<f32>, expected: &Array2<f32>) -> f32{
        match self {
            // MSE(xᵢ,yᵢ) = 1/n Σ(i=0;n) (yᵢ-ŷᵢ)^2
            Cost::MSE => { 
                let mut error: Array2<f32> = expected - predicted;
                let squared_error: &Array2<f32> = power_of(&mut error, 2);
                squared_error.sum() / squared_error.len() as f32
            },
            //max(x, 0) - x * z + log(1 + exp(-abs(x)))
            Cost::CategoricalCrossEntropy => {
                let eps: f32 = 1e-15;
                let no_zeros = &mut predicted.map(|x| if *x > 1. {1.-eps} else if *x < 0. {eps} else {*x});
                -(expected * &log_weight(no_zeros).to_owned() - (1. - expected).to_owned() * &log_weight(&mut (1. - &no_zeros.clone())).to_owned()).sum()
            }
        }
    }
    
    pub fn calculate_batch(&self, predicted: &ForwardBatch, expected: &Vec<Array2<f32>>) -> f32{
        match self {
            // MSE(xᵢ,yᵢ) = 1/n Σ(i=0;n) (yᵢ-ŷᵢ)^2
            Cost::MSE => { 
                let mut outp: f32 = 0f32;
                let batch_size = predicted.len();
                for i in 0..batch_size {
                    outp += Self::calculate(&self, predicted[i][1].last().unwrap(), &expected[i])
                }
                outp / batch_size as f32
            },
            Cost::CategoricalCrossEntropy => {
                let mut outp = 0.;
                let batch_size = predicted.len();
                for i in 0..batch_size {
                    outp += self.calculate(&predicted[i][1].last().unwrap(), &expected[i]);
                }
                outp / batch_size as f32
            }
        }
    }

    pub fn derivate(&self, predicted: &Array2<f32>, expected: &Array2<f32>) -> f32{
        match self {
            // ∂MSE/∂ŷ = -2/n Σ(i=0;n) (yᵢ-ŷᵢ)
            Cost::MSE => {
                let error: Array2<f32> = -2. * (expected - predicted);
                &error.sum() / error.len() as f32
            },
            Cost::CategoricalCrossEntropy => {
                let eps: f32 = 1e-15;
                let no_zeros = &predicted.map(|x| if *x > 1. {1.-eps} else if *x < 0. {eps} else {*x});
                -((no_zeros / expected) + (1. + no_zeros) / (1. - no_zeros)).sum()
            }
        }
    }

    pub fn derivate_batch(&self, predicted: &ForwardBatch, expected: &Vec<Array2<f32>>) -> Array2<f32> {
        match self {
            // ∂MSE/∂ŷ = -2/n Σ(i=0;n) (yᵢ-ŷᵢ)
            Cost::MSE => {
                let m: &[usize] = expected[0].shape();
                let mut outp: Array2<f32> = create_weight(&vec![m[0],m[1]]);
                let batch_size = predicted.len();
                for i in 0..batch_size {
                    // using predicted - expected as a substitute for -(expected - predicted)
                    outp.add_assign(self.calculate(predicted[i][1].last().unwrap(), &expected[i]));
                }
                scalar_div(&mut outp, batch_size as f32).to_owned()
            },
            Cost::CategoricalCrossEntropy => {
                let m: &[usize] = expected[0].shape();
                let outp: &mut Array2<f32> = &mut create_weight(&vec![m[0],m[1]]);
                let batch_size = predicted.len();
                for i in 0..batch_size {
                    outp.add_assign(self.calculate(predicted[i][1].last().unwrap(), &expected[i]));
                }
                scalar_div(outp, batch_size as f32).to_owned()
            }
        }
    }
}
