mod activations;
mod layers;
mod matrixutil;
mod netutil;
mod cost;
mod datasets;
mod typings;
mod optimizers;
use crate::activations::Activations;
use crate::layers::Layers::Dense;
use crate::netutil::{Sequential, Net};
use crate::optimizers::Optimizers;
use crate::cost::Cost;
use crate::datasets::mnist_loader;
use crate::typings::Sample;
fn main() {
    
    //load the dataset
    let mut dataset: Vec<Sample> = Vec::new();

    dataset = mnist_loader(dataset, 1);

    let epochs = 250;
    let learning_rate: f32 = 5e-3;
    let first_sample: Sample = dataset[0].clone();
    //let second_sample: Sample = dataset[1].clone();
    let batch_size: usize = 128;

    let input_dim = 784;
    let mut model = Sequential::new(input_dim.clone(), Cost::MSE);
    model.add(Dense{units: 128, activation: Activations::ReLU, init_func: String::from("he")});
    model.add(Dense{units: 32, activation: Activations::ReLU, init_func: String::from("he")});
    model.add(Dense{units: 10, activation: Activations::Softmax, init_func: String::from("he")});
    model.summary();

    model.train(dataset.clone(), Optimizers::SGD, learning_rate, batch_size, epochs);
    let pred_one = model.predict(first_sample.0);
    //let pred_two = model.predict(second_sample.0);

    println!("{}, {}", pred_one, first_sample.1);
    //println!("{}, {}", pred_two, second_sample.1);
}