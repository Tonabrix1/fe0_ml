#![allow(unused_variables, unused_imports)]
mod activations;
mod layers;
mod matrixutil;
mod netutil;
mod cost;
mod datasets;
mod typings;
mod optimizers;
mod unit_tests;
use crate::{activations::Activations::{LeakyReLU, Softmax}, layers::Layers::Dense, netutil::{Sequential, Net}, 
            optimizers::Optimizers, cost::Cost::MSE, datasets::mnist_loader, typings::Sample};
use ndarray::Array2;
fn main() {
    
    //load the dataset
    let mut dataset: Vec<Sample> = Vec::new();

    dataset = mnist_loader(dataset, 1);

    let epochs = 4000;
    let learning_rate: f32 = 1e-1;
    let first_sample: &Sample = &dataset[0];
    let batch_size: usize = 32;

    let input_dim = 784;
    let mut model = Sequential::new(input_dim, MSE);
    model.add(Dense{units: 128, activation: LeakyReLU{a: 0.02}, init_func: "he".to_string()});
    model.add(Dense{units: 32, activation: LeakyReLU{a: 0.02}, init_func: "he".to_string()});
    model.add(Dense{units: 10, activation: Softmax, init_func: "glorot".to_string()});
    //model.summary();

    //model.train(dataset.clone(), Optimizers::SGD{momentum: 0.9}, learning_rate, batch_size, epochs);
    //let pred_one = model.predict(&first_sample.0);

    //println!("{}, {}", pred_one, first_sample.1);
    unit_tests::leakyrelu_unit_tests();

}