mod activations;
mod layers;
mod matrixutil;
mod netutil;
use crate::activations::{Activation, LeakyReLU, ReLU, Sigmoid, Softmax, ELU, GELU, SELU};
use crate::layers::{Dense, Layer};
use crate::matrixutil::{create_layer, rand_layer};

fn main() {
    let mut l1 = Dense::new((256, 256), Softmax.new(), None);
    println!("{:?}", l1.get_weights());
    l1.activate_mut();
    println!("{:?}", l1.get_weights().sum())
}
