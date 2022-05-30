use ndarray::Array2;
use crate::matrixutil::{create_layer, init_he, init_xavier, rand_layer};
use crate::activations::{Activation};

// struct that holds each layer
pub trait Layer{
    fn get_weights(&self) -> Array2<f32>;
}

pub struct Dense {
    weights: Array2<f32>,
    activation: Box<dyn Activation>,
}

impl Dense{
    pub fn new(
        dim: (usize, usize),
        activation: Box<dyn Activation>,
        init_func: Option<String>,
    ) -> Dense {
        Dense {
            weights: match &init_func.unwrap_or(String::new()).to_lowercase()[..] {
                "xavier" | "glorot" => init_xavier(create_layer(dim.0, dim.1), dim.0),
                "kaiming" | "he" => init_he(create_layer(dim.0, dim.1), dim.0),
                //"lecun" => init_lecun(create_layer(dim.0, dim.1), dim.0)
                _ => rand_layer(create_layer(dim.0, dim.1), -1., 1.),
            },
            activation: activation,
        }
    }

    pub fn activate(&self) -> Array2<f32> {
        self.activation.activate(self.weights.clone())
    }

    pub fn activate_mut(&mut self) {
        self.weights = self.activation.activate(self.weights.clone());
    }
}

impl Layer for Dense {
    fn get_weights(&self) -> Array2<f32> {
        self.weights.clone()
    }
}
