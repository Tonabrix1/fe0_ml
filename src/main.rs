mod activations;
mod layers;
mod matrixutil;
mod netutil;
mod cost;
use crate::activations::{Activations};
use crate::layers::Layers::{Dense};
use crate::netutil::{Sequential, Net, Sample};
use crate::matrixutil::{init_rand, create_layer};
use crate::cost::{Cost};
use ndarray::{Array2,Ix2};

fn main() {
    let input_dim = 784;
    let mut model = Sequential::new(input_dim.clone(), Cost::MSE);
    model.add(Dense{units: 128, activation: Activations::ReLU, init_func: String::from("glorot")});
    model.add(Dense{units: 32, activation: Activations::ReLU, init_func: String::from("glorot")});
    model.add(Dense{units: 10, activation: Activations::Softmax, init_func: String::from("glorot")});
    model.summary();

    let input = init_rand::<Ix2>(vec![1,input_dim]);
    let mut output: Array2<f32> = create_layer(vec![1,10]);
    output[[0,0]] = 1f32;
    let sample = Sample(input.clone(), output);
    let epochs = 2;
    let learning_rate = 0.001;
    for i in 0..epochs {
        model.train(vec![sample.clone()], learning_rate);
    }
    model.predict(sample.0);
}