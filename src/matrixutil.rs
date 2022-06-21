use ndarray::{Array2, ArrayBase, Array, ArrayView, DataMut, Dimension, OwnedRepr};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

//initializes a tensor of rank dim.len() where every element is a 0
pub fn create_layer<D>(dim: Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    let mut shape = D::zeros(dim.len());
    shape.as_array_view_mut().assign(&ArrayView::from(&dim));
    Array::zeros(shape)
}

// replaces all values in a 2D matrix with a random number between x and y (inclusive)
// layer : the original 2D matrix
// out : the transformed 2D matrix where all values have been randomized
pub fn rand_layer<S,D>(mut layer: ArrayBase<S, D>, x: f32, y: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    layer.mapv_inplace(|_| thread_rng().gen_range(x..y));
    layer
}

// replaces all values in a 2D matrix with a random number taken from a gaussian (normal) distribution
pub fn rand_gaussian<D>(mut layer: ArrayBase<OwnedRepr<f32>, D>, mean: f32, std: f32) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    let norm = Normal::new(mean, std).unwrap();
    let mut gen = thread_rng();
    layer.mapv_inplace(|_| norm.sample(&mut gen));
    layer
}

//simply creates a layer and initializes each value to a number between 1 and -1
pub fn init_rand<D>(dim: Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    rand_layer(create_layer(dim), -1., 1.)
}

// xavier glorot's initialization function, used for tanh and sigmoid activations
// num_nodes : the number of input nodes (weights matrix.shape[1])]
// uses usize so you can directly pass .shape()[0] to it
pub fn init_xavier<D>(dim: Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    let layer = rand_layer(create_layer(dim.clone()), 0., 1.);
    let upper: f32 = 1. / (dim.clone()[0] as f32).sqrt();
    let lower: f32 = -1. / (dim[0] as f32).sqrt();
    scalar_mult(scalar_add(layer, lower), upper - lower)
}

// kaiming he's initialization function, used for ReLU activation
// num_nodes : the number of input nodes (weights matrix.shape[1])
pub fn init_he<D>(dim: Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    let layer = rand_layer(create_layer(dim.clone()), 0., 1.);
    let std: f32 = (2. / (dim[0] as f32)).sqrt();
    scalar_mult(rand_gaussian(layer, 0., 1.), std)
}

// calculates the exponential of a tensor
pub fn exp_layer<S,D>(mut layer: ArrayBase<S, D>) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    layer.mapv_inplace(|x: f32| x.exp());
    layer
}

// divides each value in a 2D array by val
pub fn scalar_div<S,D>(mut layer: ArrayBase<S, D>, val: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    layer.mapv_inplace(|x: f32| x / val);
    layer
}

// multiplies each element in a 2D array by val
pub fn scalar_mult<S,D>(mut layer: ArrayBase<S, D>, val: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    layer.mapv_inplace(|x: f32| x * val);
    layer
}

// subtracts val from each value in a 2D array
pub fn scalar_sub<S,D>(mut layer: ArrayBase<S, D>, val: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    layer.mapv_inplace(|x: f32| x - val);
    layer
}

// adds val to each value in an array
pub fn scalar_add<S,D>(mut layer: ArrayBase<S, D>, val: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    layer.mapv_inplace(|x: f32| x + val);
    layer
}

// replaces each value in a 2D array with its reciprocal (1/value)
pub fn scalar_reciprocal<S,D>(mut layer: ArrayBase<S, D>) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    layer.mapv_inplace(|x: f32| 1. / x);
    layer
}

// raises each value in a 2D array to the power of val
pub fn power_of<S,D>(mut layer: ArrayBase<S, D>, val: i32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    layer.mapv_inplace(|x: f32| x.powi(val));
    layer
}

// finds the largest element in an 2D-array and returns the index of it as a tuple
pub fn arg_max(layer: Array2<f32>) -> (usize, usize) {
    let mut max_indx: (usize, usize) = (0, 0);
    let mut max: f32 = layer[[0, 0]];
    let m = layer.shape();
    for i in 0..m[0] {
        for j in 0..m[1] {
            let v = layer[[i, j]];
            if v > max {
                max = v;
                max_indx = (i, j);
            }
        }
    }
    max_indx
}
/*
// preforms matrix multiplication of two arrays (A, B) and returns a new array with dimensions (A.shape[0],B.shape[1])
pub fn mat_mul(arr1: Array2<f32>, arr2: Array2<f32>) -> Array2<f32> {
    let m1: usize = arr1.shape()[0];
    let m2: &[usize] = arr2.shape();
    let mut outp: Array2<f32> = create_layer(m1, m2[1]);
    for i in 0..m1 {
        for j in 0..m2[1] {
            for k in 0..m2[0] {
                outp[[i, j]] = arr1[[i, k]] * arr2[[k, j]];
            }
        }
    }
    outp
}
*/