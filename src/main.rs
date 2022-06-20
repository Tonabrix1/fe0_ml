mod activations;
mod layers;
mod matrixutil;
mod netutil;
use crate::activations::{Activations};
use crate::layers::{Dense, Layer};
use crate::matrixutil::{init_rand, create_layer};
use crate::netutil::{Sequential, Net};

fn main() {
    let mut model = Sequential::new();
    model.add(Dense::new(10, 20, Activations::Softmax, Some("glorot")));

    let mut l1 = &mut model.layers[0];
    println!("initalized weights:\n{:?}", l1.get_weights());

    l1.forward_propagate(&init_rand(20,1));
    println!("weights after forward prop: {:?}", l1.get_weights());
}


/*fn test<S,D>(mut x: ArrayBase<S, D>) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    x.mapv_inplace(|x| x + 1.);
    x
}*/
