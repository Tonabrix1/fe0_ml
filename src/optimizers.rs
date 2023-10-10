use crate::{matrixutil::transpose, netutil::Sequential, typings::ForwardBatch};
use ndarray::Array2;

pub enum Optimizers {
    SGD,
}

impl Optimizers {
    pub fn backward(
        &self,
        network: &Sequential,
        predictions: &ForwardBatch,
        input: &Vec<Array2<f32>>,
        expected: &Vec<Array2<f32>>,
    ) -> Vec<Vec<Array2<f32>>> {
        match self {
            Optimizers::SGD => {
                let batch_size: usize = predictions.len();
                let last_pred: usize = predictions[0][0].len();
                let mut c_wrt_z: Array2<f32> = Array2::<f32>::zeros((1, 1));
                let mut c_wrt_a: Array2<f32>;
                let mut weight_updates: Vec<Array2<f32>> = Vec::new();
                let mut bias_updates: Vec<Array2<f32>> = Vec::new();

                //TODO: store transposes of the weights every batch to avoid creating a new transpose array for every batch

                for j in 0..batch_size {
                    for i in (0..last_pred).rev() {
                        let mut z = &predictions[j][0][i];
                        let a_prev = if i > 0 {
                            &predictions[j][1][i - 1]
                        } else {
                            &input[j]
                        };
                        // ∂C/∂w = ∂Z/∂w * ∂A/∂Z * ∂C/∂A
                        if i == last_pred - 1 {
                            // ∂C/∂zₙ = ∂aₙ/∂zₙ * ∂C/∂aₙ
                            c_wrt_z = network.layers[i].derivate_activation(&mut z)
                                * network.cost.derivate(&predictions, &expected);
                        } else {
                            // ∂C/∂aₙ₋₁ = ∂zₙ/∂aₙ₋₁ * ∂C/∂zₙ
                            // ∂zₙ/∂aₙ₋₁ = wₙ.T
                            c_wrt_a = c_wrt_z.dot(&transpose(&network.weights[i + 1]));
                            // ∂C/∂zₙ₋₁ = ∂aₙ₋₁/∂zₙ₋₁ * ∂C/∂aₙ₋₁
                            c_wrt_z = network.layers[i].derivate_activation(z) * c_wrt_a;
                        }

                        // on the first batch, fill the vectors
                        if j == 0 {
                            // ∂C/∂bₙ = ∂Z/∂bₙ * ∂A/∂Z * ∂C/∂A
                            // ∂Z/∂bₙ = 1
                            // ∂C/∂bₙ = 1 * c_wrt_z = c_wrt_z
                            bias_updates.push(c_wrt_z.clone());
                            // ∂C/∂wₙ = ∂zₙ/∂wₙ * c_wrt_z
                            // Z(w,X,b) = w.X + b
                            // ∂Z/∂w = Xᵀ
                            weight_updates.push(transpose(a_prev).dot(&c_wrt_z));
                        } else {
                            let curr = last_pred - (1usize + i);
                            // on batches after the first, accumulate updates
                            bias_updates[curr] = &bias_updates[curr] + &c_wrt_z;
                            weight_updates[curr] =
                                &weight_updates[curr] + &transpose(a_prev).dot(&c_wrt_z);
                        }
                    }
                }
                vec![weight_updates, bias_updates]
            }
        }
    }
}
