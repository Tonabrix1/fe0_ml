mod activations;
mod layers;
mod matrixutil;
mod netutil;
use ndarray::{IxDyn,OwnedRepr,ArrayBase};
use crate::activations::{Activations};
use crate::layers::{Dense};
use crate::matrixutil::{init_rand, create_layer};
use crate::netutil::{Sequential, Net};

fn main() {
    let mut model = Sequential::new();
    model.add(Dense::new(vec![10, 20, 2, 2, 2], Activations::Softmax, Some("glorot")));

    let l1 = &mut model.layers[0];
    println!("initalized weights:\n{:?}", l1.get_weights());

    //l1.forward_propagate(&init_rand(vec![1,20]));
    //println!("weights after forward prop: {:?}", l1.get_weights());
}