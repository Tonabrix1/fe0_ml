use ndarray::Array2;
use crate::matrixutil::{init_he, init_xavier, init_rand, create_layer};
use crate::activations::{Activations};

// struct that holds each layer
pub trait Layer{
    fn get_weights(&self) -> Array2<f32>;
    fn forward_propagate(&mut self, prev: &Array2<f32>);
}

pub struct Dense {
    weights: Array2<f32>,
    bias: Array2<f32>,
    activation: Activations,
}

impl Dense{
    pub fn new(
        dim1: usize,
        dim2: usize,
        activation: Activations,
        init_func: Option<&str>,
    ) -> Box<dyn Layer> {
        let new_weights: Array2<f32> = match &init_func.unwrap_or("").to_string().to_lowercase()[..] {
            "xavier" | "glorot" => init_xavier(dim1, dim2),
            "kaiming" | "he" => init_he(dim1, dim2),
            //"lecun" => init_lecun(dim1, dim2)
            _ => init_rand(dim1, dim2),
        };
        let weights_shape: usize = new_weights.shape()[0];
        Box::new(Dense {
            // converting the string back and forth like this is ugly as fuck
            weights: new_weights,
            /* https://cs231n.github.io/neural-networks-2/ : It is possible and common to initialize the biases to be zero,
             * since the asymmetry breaking is provided by the small random numbers in the weights.
             * For ReLU non-linearities, some people like to use small constant value such as 0.01 for all biases
             * because this ensures that all ReLU units fire in the beginning and therefore obtain and propagate some gradient.
             * However, it is not clear if this provides a consistent improvement (in fact some results seem to indicate that this performs worse)
             * and it is more common to simply use 0.
             */
            bias: create_layer(weights_shape,1),
            activation: activation,
        })
    }
}

impl Layer for Dense {
    fn get_weights(&self) -> Array2<f32> {
        self.weights.clone()
    }

    fn forward_propagate(&mut self, prev: &Array2<f32>) {
        self.activation.activate(self.weights.dot(prev) + &self.bias);
    }
}
