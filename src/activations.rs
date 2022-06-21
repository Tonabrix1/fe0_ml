use ndarray::{
    ArrayBase, Dimension, DataMut, DataOwned, RawDataClone
};
use std::f32::consts::PI;
use crate::matrixutil::{
    exp_layer, scalar_add, scalar_sub, scalar_div
};

// enum storing each activation function
#[allow(dead_code)]
pub enum Activations {
    Sigmoid,
    ReLU,
    LeakyReLU{ a: f32 },
    Softmax,
    ELU{ a: f32 },
    SELU,
    GELU,
}

impl Activations {
    pub fn activate<S,D>(&self, mut layer: ArrayBase<S, D>) -> ArrayBase<S, D> where S: DataMut<Elem = f32> + RawDataClone, D: Dimension, {
        match self {
            Activations::Sigmoid => {
                layer.mapv_inplace(|x| 1. / (1. + x.exp()));
                layer
            },
            Activations::ReLU => {
                layer.mapv_inplace(|x| if x > 0. { x } else { 0. });
                layer
            },
            Activations::LeakyReLU  { a } => {
                layer.mapv_inplace(|x| if x >= 0. { x } else { x * (*a) });
                layer
            },
            Activations::Softmax => {
                let ex = exp_layer(layer);
                let sum = ex.sum();
                scalar_div(ex, sum)
            },
            Activations::ELU { a } => {
                layer.mapv_inplace(|x| if x > 0. { x } else { (x.exp() - 1.) * (*a) });
                layer
            },
            Activations::SELU => {
                let a = 1.6732632423543772848170429916717;
                let l = 1.0507009873554804934193349852946;
                layer.mapv_inplace(|x| if x > 0. { x * l } else { l * (a * x.exp() - a) });
                layer
            },
            Activations::GELU => {
                //0.5x(1+tanh(√2/π(x+0.044715x^3)))
                let c1 = (2. / PI).sqrt();
                let f1 = |x: f32| x + 0.044715 * x.powi(3);
                layer.mapv_inplace(|x| 0.5 * x * (1. + (c1 * f1(x)).tanh()));
                layer
            },
        }
    }

    // I know this isn't technically grammatically correct but I like the name for homogeneity
    pub fn derivate<S,D>(&self, mut layer: ArrayBase<S, D>) -> ArrayBase<S, D> where S: DataMut<Elem = f32> + DataOwned + RawDataClone, D: Dimension, {
        match self {
            Activations::Sigmoid => {
                let ex = exp_layer(-1f32 * layer);
                // (e^-x)+1
                let denom = scalar_add(ex.clone(), 1.);
                // e^-x/((e^-1)+1)^2 = e^-x/((e^-1)+1)*((e^-1)+1)
                ex / (denom.clone() * denom)
            },
            Activations::ReLU => {
                // technically it's undefined at x[[i,j]] == 0
                layer.mapv_inplace(|x| if x > 0. { 1. } else { 0. });
                layer
            },
            Activations::LeakyReLU { a } => {
                layer.mapv_inplace(|x| if x >= 0. { 1. } else { *a });
                layer
            },
            Activations::Softmax => {
                let sf = self.activate(layer);
                sf.clone() * scalar_sub(sf, 1.)
            },
            Activations::ELU { a } => {
                let act = |x: f32| if x > 0. { x } else { (x.exp() - 1.) * (*a) };
                layer.mapv_inplace(|x| if x > 0. { 1. } else { act(x) + (*a) });
                layer
            },
            Activations::SELU => {
                let a = 1.6732632423543772848170429916717;
                let l = 1.0507009873554804934193349852946;
                layer.mapv_inplace(|x| if x > 0. { l } else { l * (a * x.exp()) });
                layer
            },
            Activations::GELU => {
                //0.5tanh(0.0356774x^3+0.797885x)+(0.0535161x^3+0.398942x)sech^2(0.0356774x^3+0.797885x)+0.5
                let f1 = |x: f32| 0.0356774 * x.powi(3) + 0.797885 * x;
                let f2 = |x: f32| 0.0535161 * x.powi(3) + 0.398942 * x;
                let sech = |x: f32| 1. / x.cosh();
                layer.mapv_inplace(|x| 0.5 * f1(x).tanh() + f2(x) * sech(f1(x)).powi(2) + 0.5);
                layer
            },
        }
    }
}
