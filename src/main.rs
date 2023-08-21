mod activations;
mod cost;
mod datasets;
mod layers;
mod matrixutil;
mod netutil;
mod optimizers;
mod typings;
use crate::{
    activations::Activations::{ReLU, Softmax},
    cost::Cost::MSE,
    datasets::mnist_loader,
    layers::Layers::Dense,
    netutil::{Net, Sequential},
    optimizers::Optimizers,
    typings::Sample,
};
fn main() {
    //load the dataset
    let mut dataset: Vec<Sample> = Vec::new();

    dataset = mnist_loader(dataset, 1000);

    let epochs = 250;
    let learning_rate: f32 = 5e-3;
    let first_sample: &Sample = &dataset[0];
    let batch_size: usize = 128;

    let input_dim = 784;
    let mut model = Sequential::new(input_dim, MSE);
    model.add(Dense {
        units: 128,
        activation: ReLU,
        init_func: String::from("he"),
    });
    model.add(Dense {
        units: 32,
        activation: ReLU,
        init_func: String::from("he"),
    });
    model.add(Dense {
        units: 10,
        activation: Softmax,
        init_func: String::from("he"),
    });
    model.summary();

    model.train(
        dataset.clone(),
        Optimizers::SGD,
        learning_rate,
        batch_size,
        epochs,
    );
    let pred_one = model.predict(&first_sample.0);

    println!("{}, {}", pred_one, first_sample.1);
}
