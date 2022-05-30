use ndarray::Array2;
use std::f32::consts::PI;
use crate::matrixutil::{
    create_layer, exp_layer, rand_layer, scalar_add, scalar_reciprocal,
    scalar_sub,
};

// trait used to implement activation and derivative functions for each activation function
pub trait Activation {
    fn new(&self) -> Box<dyn Activation>;
    fn activate(&self, layer: Array2<f32>) -> Array2<f32>;
    // I know this technically isn't correct grammatically but I like it for homogeneity
    fn derivate(&self, layer: Array2<f32>) -> Array2<f32>;
    // only available for activation functions involving and alpha value
    fn activate_a(&self, layer: Array2<f32>, a: f32) -> Array2<f32> {
        panic!("THIS IS AN UNIMPLEMENTED METHOD...");
    }
    // only available for activation functions involving and alpha value
    fn derivate_a(&self, layer: Array2<f32>, a: f32) -> Array2<f32> {
        panic!("THIS IS AN UNIMPLEMENTED METHOD...");
    }
    // only available for activation functions involving and alpha and lambda values
    fn activate_al(&self, layer: Array2<f32>, a: f32, l: f32) -> Array2<f32> {
        panic!("THIS IS AN UNIMPLEMENTED METHOD...");
    }
    // only available for activation functions involving and alpha and lambda values
    fn derivate_al(&self, layer: Array2<f32>, a: f32, l: f32) -> Array2<f32> {
        panic!("THIS IS AN UNIMPLEMENTED METHOD...");
    }
}

pub struct Sigmoid;
impl Activation for Sigmoid {
    fn new(&self) -> Box<dyn Activation> {
        Box::new(Sigmoid)
    }

    // take each value in an array and scale it into a number between 0 and 1
    fn activate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        // 1/(1+(e^-x))
        layer.mapv_inplace(|x| 1. / (1. + x.exp()));
        layer
    }

    fn derivate(&self, layer: Array2<f32>) -> Array2<f32> {
        // e^-x
        let ex = exp_layer(-1. * layer);
        // (e^-x)+1
        let denom = scalar_add(ex.clone(), 1.);
        // e^-x/((e^-1)+1)^2 = e^-x/((e^-1)+1)*((e^-1)+1)
        ex / (denom.clone() * denom)
    }
}

pub struct ReLU;
impl Activation for ReLU {
    fn new(&self) -> Box<dyn Activation> {
        Box::new(ReLU)
    }

    fn activate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        layer.mapv_inplace(|x| if x > 0. { x } else { 0. });
        layer
    }

    fn derivate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        // technically it's undefined at x[[i,j]] == 0
        layer.mapv_inplace(|x| if x > 0. { 1. } else { 0. });
        layer
    }
}

pub struct LeakyReLU;
impl Activation for LeakyReLU {
    // alpha is set to 0.01 by default
    fn new(&self) -> Box<dyn Activation> {
        Box::new(LeakyReLU)
    }

    fn activate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        let a = 0.01;
        layer.mapv_inplace(|x| if x >= 0. { x } else { x * a });
        layer
    }

    fn derivate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        let a = 0.01;
        layer.mapv_inplace(|x| if x >= 0. { 1. } else { a });
        layer
    }

    fn activate_a(&self, mut layer: Array2<f32>, a: f32) -> Array2<f32> {
        layer.mapv_inplace(|x| if x >= 0. { x } else { x * a });
        layer
    }

    fn derivate_a(&self, mut layer: Array2<f32>, a: f32) -> Array2<f32> {
        layer.mapv_inplace(|x| if x >= 0. { 1. } else { a });
        layer
    }
}

pub struct Softmax;
impl Activation for Softmax {
    fn new(&self) -> Box<dyn Activation> {
        Box::new(Softmax)
    }

    // creates a matrix of probabilities summing to 1 (or at least close enough :^)
    fn activate(&self, layer: Array2<f32>) -> Array2<f32> {
        let ex = exp_layer(layer);
        ex.clone() / ex.sum()
    }

    fn derivate(&self, layer: Array2<f32>) -> Array2<f32> {
        let sf = self.activate(layer);
        sf.clone() * scalar_sub(sf, 1.)
    }
}

pub struct ELU;
impl Activation for ELU {
    // alpha is set to 0.2 by default
    fn new(&self) -> Box<dyn Activation> {
        Box::new(ELU)
    }

    fn activate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        let a = 0.2;
        layer.mapv_inplace(|x| if x > 0. { x } else { (x.exp() - 1.) * a });
        layer
    }

    // this is an ugly way to do this but there is no clean way that can be used for every activation
    // function since softmax cannot be run on a single value
    fn derivate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        let a = 0.2;
        let act = |x: f32| if x > 0. { x } else { (x.exp() - 1.) * a };
        layer.mapv_inplace(|x| if x > 0. { 1. } else { act(x) + a });
        layer
    }

    fn activate_a(&self, mut layer: Array2<f32>, a: f32) -> Array2<f32> {
        layer.mapv_inplace(|x| if x > 0. { x } else { (x.exp() - 1.) * a });
        layer
    }

    fn derivate_a(&self, mut layer: Array2<f32>, a: f32) -> Array2<f32> {
        let act = |x: f32| if x > 0. { x } else { (x.exp() - 1.) * a };
        layer.mapv_inplace(|x| if x > 0. { 1. } else { act(x) + a });
        layer
    }
}

pub struct SELU;
impl Activation for SELU {
    //TODO: implement lecun_norm initialization for SELU
    fn new(&self) -> Box<dyn Activation> {
        Box::new(SELU)
    }

    fn activate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        let a = 1.6732632423543772848170429916717;
        let l = 1.0507009873554804934193349852946;
        layer.mapv_inplace(|x| if x > 0. { x * l } else { l * (a * x.exp() - a) });
        layer
    }

    fn derivate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        let a = 1.6732632423543772848170429916717;
        let l = 1.0507009873554804934193349852946;
        layer.mapv_inplace(|x| if x > 0. { l } else { l * (a * x.exp()) });
        layer
    }

    fn activate_a(&self, mut layer: Array2<f32>, a: f32) -> Array2<f32> {
        let l = 1.0507009873554804934193349852946;
        layer.mapv_inplace(|x| if x > 0. { x * l } else { l * (a * x.exp() - a) });
        layer
    }

    fn derivate_a(&self, mut layer: Array2<f32>, a: f32) -> Array2<f32> {
        let l = 1.0507009873554804934193349852946;
        layer.mapv_inplace(|x| if x > 0. { l } else { l * (a * x.exp()) });
        layer
    }

    fn activate_al(&self, mut layer: Array2<f32>, a: f32, l: f32) -> Array2<f32> {
        layer.mapv_inplace(|x| if x > 0. { x * l } else { l * (a * x.exp() - a) });
        layer
    }

    fn derivate_al(&self, mut layer: Array2<f32>, a: f32, l: f32) -> Array2<f32> {
        layer.mapv_inplace(|x| if x > 0. { l } else { l * (a * x.exp()) });
        layer
    }
}

pub struct GELU;
impl Activation for GELU {
    fn new(&self) -> Box<dyn Activation> {
        Box::new(GELU)
    }

    //0.5x(1+tanh(√2/π(x+0.044715x^3)))
    fn activate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        let c1 = (2. / PI).sqrt();
        let f1 = |x: f32| x + 0.044715 * x.powi(3);
        layer.mapv_inplace(|x| 0.5 * x * (1. + (c1 * f1(x)).tanh()));
        layer
    }

    //0.5tanh(0.0356774x^3+0.797885x)+(0.0535161x^3+0.398942x)sech^2(0.0356774x^3+0.797885x)+0.5
    fn derivate(&self, mut layer: Array2<f32>) -> Array2<f32> {
        let f1 = |x: f32| 0.0356774 * x.powi(3) + 0.797885 * x;
        let f2 = |x: f32| 0.0535161 * x.powi(3) + 0.398942 * x;
        let sech = |x: f32| 1. / x.cosh();
        layer.mapv_inplace(|x| 0.5 * f1(x).tanh() + f2(x) * sech(f1(x)).powi(2) + 0.5);
        layer
    }
}
