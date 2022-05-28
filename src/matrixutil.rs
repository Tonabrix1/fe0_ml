use ndarray::{s, Array2};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

pub fn create_layer(dim1: usize, dim2: usize) -> Array2<f32> {
    //initializes an array of 0's: dim1=1; dim2=3
    // [[0, 0, 0]]
    //the first element is accessed with Array2[[0,0]]
    let out = Array2::<f32>::zeros((dim1, dim2));
    out
}

//this is a deprecated method just awaiting a rewrite
pub fn generate_bias_layer(
    layer1: Array2<f32>,
    layer2: Array2<f32>,
    min: Option<f32>,
    max: Option<f32>,
) -> Array2<f32> {
    let temp = create_layer(layer1.shape()[0], layer2.shape()[1]);
    rand_layer(temp, min.unwrap_or(0.), max.unwrap_or(1.))
}

// replaces all values in a 2D matrix with a random number between x and y (inclusive)
// layer : the original 2D matrix
// out : the transformed 2D matrix where all values have been randomized
pub fn rand_layer(mut layer: Array2<f32>, x: f32, y: f32) -> Array2<f32> {
    layer.mapv_inplace(|_| thread_rng().gen_range(x..y));
    layer
}

// replaces all values in a 2D matrix with a random number taken from a gaussian (normal) distribution
pub fn rand_gaussian(mut layer: Array2<f32>, mean: f32, std: f32) -> Array2<f32> {
    let norm = Normal::new(mean, std).unwrap();
    let mut gen = thread_rng();
    layer.mapv_inplace(|_| norm.sample(&mut gen));
    layer
}

// xavier glorot's initialization function, used for tanh and sigmoid activations
// num_nodes : the number of input nodes (weights matrix.shape[1])]
// uses usize so you can directly pass .shape()[0] to it
pub fn init_xavier(layer: Array2<f32>, num_nodes: usize) -> Array2<f32> {
    let upper: f32 = 1. / (num_nodes as f32).sqrt();
    let lower: f32 = -1. / (num_nodes as f32).sqrt();
    scalar_mult(scalar_add(rand_layer(layer, 0., 1.), lower), upper - lower)
}

// kaiming he's initialization function, used for ReLU activation
// num_nodes : the number of input nodes (weights matrix.shape[1])
pub fn init_he(mut layer: Array2<f32>, num_nodes: usize) -> Array2<f32> {
    let std: f32 = (2. / (num_nodes as f32)).sqrt();
    scalar_mult(rand_gaussian(layer, 0., 1.), std)
}

// calculates the exponential of a matrix
// layer : the original 2D matrix
// out : the transformed 2D matrix where each value-x has been replaced by e^x
pub fn exp_layer(mut layer: Array2<f32>) -> Array2<f32> {
    layer.mapv_inplace(|x| x.exp());
    layer
}

// divides each value in a 2D array by val
pub fn scalar_div(mut layer: Array2<f32>, val: f32) -> Array2<f32> {
    layer.mapv_inplace(|x| x / val);
    layer
}

// multiplies each element in a 2D array by val
pub fn scalar_mult(mut layer: Array2<f32>, val: f32) -> Array2<f32> {
    layer.mapv_inplace(|x| x * val);
    layer
}

// subtracts val from each value in a 2D array
pub fn scalar_sub(mut layer: Array2<f32>, val: f32) -> Array2<f32> {
    layer.mapv_inplace(|x| x - val);
    layer
}

// calls scalar_sub and multiplies val by -1.0
pub fn scalar_add(mut layer: Array2<f32>, val: f32) -> Array2<f32> {
    layer.mapv_inplace(|x| x + val);
    layer
}

// replaces each value in a 2D array with it's reciprocal (1/value)
pub fn scalar_reciprocal(mut layer: Array2<f32>) -> Array2<f32> {
    layer.mapv_inplace(|x| 1. / x);
    layer
}

// raises each value in a 2D array to the power of val
pub fn power_of(mut layer: Array2<f32>, val: i32) -> Array2<f32> {
    layer.mapv_inplace(|x| x.powi(val));
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

// preforms matrix multiplication of two arrays (A, B) and returns a new array with dimensions (A.shape[0],B.shape[1])
pub fn mat_mul(arr1: Array2<f32>, arr2: Array2<f32>) -> Array2<f32> {
    let m1 = arr1.shape()[0];
    let m2 = arr2.shape();
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
