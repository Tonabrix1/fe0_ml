use ndarray::Array2;
use crate::activations::Activation;
use crate::layers::Layer;

// main struct that holds a reference to the layers and biases
pub struct Net {
    pub layers: Vec<Box<dyn Layer>>,
    pub biases: Vec<Array2<f32>>,
}

// holds a single (input : onehot_label) pair,
pub struct Sample(Array2<f32>, Array2<f32>);
