use ndarray::{ArrayBase, OwnedRepr, IxDyn};
use crate::matrixutil::{init_he, init_xavier, init_rand, create_layer};
use crate::activations::{Activations};

// struct that holds each layer
pub trait Layer{
    fn get_weights(&self) -> ArrayBase<OwnedRepr<f32>, IxDyn>;
    fn forward_propagate(&mut self, prev: &ArrayBase<OwnedRepr<f32>, IxDyn>);
}

pub struct Dense {
    weights: ArrayBase<OwnedRepr<f32>, IxDyn>,
    bias: ArrayBase<OwnedRepr<f32>, IxDyn>,
    activation: Activations,
}

impl Dense{
    pub fn new(
        dim: Vec<usize>,
        activation: Activations,
        init_func: Option<&str>,
    ) -> Box<dyn Layer> {
        let new_weights: ArrayBase<OwnedRepr<f32>, IxDyn> = match &init_func.unwrap_or("").to_string().to_lowercase()[..] {
            "xavier" | "glorot" => init_xavier(dim.clone()),
            "kaiming" | "he" => init_he(dim.clone()),
            //"lecun" => init_lecun(dim)
            _ => init_rand(dim.clone()),
        };
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
            bias: create_layer(vec![*&dim[dim.len()-1],1]),
            activation: activation,
        })
    }
}

impl Layer for Dense {
    fn get_weights(&self) -> ArrayBase<OwnedRepr<f32>, IxDyn> {
        self.weights.clone()
    }

    fn forward_propagate(&mut self, prev: &ArrayBase<OwnedRepr<f32>, IxDyn>) {
        //self.activation.activate(self.weights.dot(prev) + &self.bias);
        //Implement for ward prop for n-dimensional tensors
    }
}
