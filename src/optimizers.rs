#![allow(dead_code)]
use ndarray::Array2;
use std::ops::{Mul};
use crate::{typings::ForwardBatch, netutil::Sequential, matrixutil::{transpose, create_weight}};

pub enum Optimizers {
    SGD{momentum: f32},
}

impl Optimizers {
    pub fn backward(&self, network: &Sequential, predictions: &ForwardBatch, input: &Vec<Array2<f32>>, expected: &Vec<Array2<f32>>, lr: f32) -> Vec<Vec<Array2<f32>>>{
        match self {
            Optimizers::SGD{momentum} => {
                let batch_size: usize = predictions.len();
                let num_layers: usize = predictions[0][0].len();
                let mut c_wrt_z: Array2<f32> = Array2::<f32>::zeros((1,1));
                let mut c_wrt_a: Array2<f32>;
                let mut weight_updates: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
                let mut bias_updates: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
                let mut velocities: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
                //TODO: store transposes of the weights every batch to avoid creating a new transpose array for every batch

                for j in 0..batch_size {
                    for i in (0..num_layers).rev() {
                        let mut z = &predictions[j][0][i];
                        if j == 0 {
                            let z_shape = z.shape();
                            velocities.push(create_weight(&vec![z_shape[0], z_shape[1]]));
                        }
                        let a_prev = if i > 0 {&predictions[j][1][i-1]} else {&input[j]};
                        // ∂C/∂w = ∂Z/∂w * ∂A/∂Z * ∂C/∂A
                        if i == num_layers - 1{
                            // ∂C/∂zₙ = ∂aₙ/∂zₙ * ∂C/∂aₙ
                            c_wrt_z = network.layers[i].derivate_activation(&mut z) * network.cost.derivate(predictions[j][1].last().unwrap(), &expected[j]);
                        } else {
                            // ∂C/∂aₙ₋₁ = ∂zₙ/∂aₙ₋₁ * ∂C/∂zₙ
                            // ∂zₙ/∂aₙ₋₁ = wₙ.T
                            c_wrt_a = c_wrt_z.dot(&transpose(&network.weights[i+1]));
                            // ∂C/∂zₙ₋₁ = ∂aₙ₋₁/∂zₙ₋₁ * ∂C/∂aₙ₋₁
                            c_wrt_z = network.layers[i].derivate_activation(z) * c_wrt_a;
                        }
                        //println!("Gradient: {:?}\n", c_wrt_z);
                        velocities[(num_layers-1) - i] = *momentum*&velocities[(num_layers-1) - i] - lr * &c_wrt_z;
                        //println!("Velocity: {:?}\n\n", velocities[(num_layers-1) - i]);
                        // ∂C/∂bₙ = ∂Z/∂bₙ * ∂A/∂Z * ∂C/∂A
                        // ∂Z/∂bₙ = 1
                        // ∂C/∂bₙ = 1 * c_wrt_z = c_wrt_z
                        // println!("update #{}: {:?}\n\n",num_layers - i, velocities[(num_layers-1) - i]);
                        bias_updates.push(velocities[(num_layers-1) - i].clone());
                        // ∂C/∂wₙ = ∂zₙ/∂wₙ * c_wrt_z
                        // Z(w,X,b) = w.X + b
                        // ∂Z/∂w = Xᵀ
                        weight_updates.push(transpose(a_prev).dot(&velocities[(num_layers-1) - i]));
                    }
                }
                vec![weight_updates, bias_updates]
            }

        }
        
    }
}