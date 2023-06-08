mod activations;
mod layers;
mod matrixutil;
mod netutil;
mod cost;
mod datasets;
mod typings;
use crate::activations::Activations;
use crate::layers::Layers::Dense;
use crate::netutil::{Sequential, Net};
use crate::cost::Cost;
use crate::datasets::mnist_loader;
use crate::typings::Sample;
fn main() {
    
    //load the dataset
    let mut dataset: Vec<Sample> = Vec::new();

    dataset = mnist_loader(dataset, 1);

    let epochs = 10000;
    let learning_rate = 0.001;
    let first_sample = dataset[0].clone();
    let batch_size: usize = 32;

    let input_dim = 784;
    let mut model = Sequential::new(input_dim.clone(), Cost::MSE);
    model.add(Dense{units: 128, activation: Activations::ReLU, init_func: String::from("he")});
    model.add(Dense{units: 32, activation: Activations::ReLU, init_func: String::from("he")});
    model.add(Dense{units: 10, activation: Activations::Softmax, init_func: String::from("he")});
    model.summary();

    model.train(dataset.clone(), learning_rate, batch_size, epochs);
    model.predict(first_sample.0);
}