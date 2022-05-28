mod matrixutil;
mod netutil;
use crate::matrixutil::{create_layer, rand_layer};
use crate::netutil::{Activation, Layer, LeakyReLU, ReLU, Sigmoid, Softmax, ELU, GELU, SELU};

fn main() {
    let l1 = Layer::new((256, 256), GELU.new(), None);
    println!("{:?}", l1.weights);
    let outp = l1.activation.activate(l1.weights);
    println!("{:?}", outp);
}
