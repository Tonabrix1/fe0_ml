mod activations;
mod layers;
mod matrixutil;
mod netutil;
use crate::activations::{Activation, LeakyReLU, ReLU, Sigmoid, Softmax, ELU, GELU, SELU};
use crate::layers::{Dense, Layer};
use crate::matrixutil::{create_layer, rand_layer};
use crate::netutil::{Sequential};

fn main() {
    let mut model = Sequential::new();
    model.add(Box::new(Dense::new((1048, 1048), Softmax.new(), None)));
    let mut l1 = Dense::new((2096, 2096), Softmax.new(), None);
    println!("initalized weights:\n{:?}", l1.get_weights());
    l1.activate_mut();
    println!("after softmax:\n{:?}", l1.get_weights());
    println!("sum of confidence matrix (should be ~1.0): {:?}", l1.get_weights().sum())
}
