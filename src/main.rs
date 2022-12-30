mod activations;
mod layers;
mod matrixutil;
mod netutil;
use crate::activations::{Activations};
use crate::layers::Layers::{Dense};
use crate::netutil::{Sequential, Net, Sample};
use crate::matrixutil::{init_rand};
use ndarray::{Array2,Ix2};

fn main() {
    let input_dim = 784;
    let mut model = Sequential::new(input_dim.clone());
    model.add(Dense{units: 128, activation: Activations::ReLU, init_func: String::from("glorot")});
    model.add(Dense{units: 32, activation: Activations::ReLU, init_func: String::from("glorot")});
    model.add(Dense{units: 10, activation: Activations::Softmax, init_func: String::from("glorot")});
    model.summary();

    let input = init_rand::<Ix2>(vec![1,input_dim]);
    println!("input: {:?}", input.clone());
    output[0] = 1;
    let sample = Sample(input, Array2::from(output));
    model.train(vec![sample]);
}