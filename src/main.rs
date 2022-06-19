mod activations;
mod layers;
mod matrixutil;
mod netutil;
use crate::activations::{Activation, LeakyReLU};
use crate::layers::{Dense, Layer};
use crate::matrixutil::init_rand;
use crate::netutil::{Sequential, Net};

fn main() {
    let mut model = Sequential::new();
    model.add(Dense::new(10, 20, LeakyReLU.new(), Some("glorot")));

    let mut l1 = &mut model.layers[0];
    println!("initalized weights:\n{:?}", l1.get_weights());

    l1.forward_propagate(&init_rand(20,1));
    println!("weights after forward prop: {:?}", l1.get_weights())
}
