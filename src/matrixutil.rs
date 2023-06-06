use ndarray::{Array2, ArrayBase, Array, ArrayView, DataMut, Dimension, OwnedRepr};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

//initializes a tensor of rank dim.len() where every element is a 0
pub fn create_weight<D>(dim: Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    let mut shape = D::zeros(dim.len());
    shape.as_array_view_mut().assign(&ArrayView::from(&dim));
    Array::zeros(shape)
}

// replaces all values in an ND matrix with a random number between x and y (inclusive)
// weight : the original ND matrix
// out : the transformed ND matrix where all values have been randomized
pub fn rand_weight<S,D>(mut weight: ArrayBase<S, D>, x: f32, y: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    weight.mapv_inplace(|_| thread_rng().gen_range(x..=y));
    weight
}

// replaces all values in an ND matrix with a random number taken from a gaussian (normal) distribution
pub fn rand_gaussian<D>(mut weight: ArrayBase<OwnedRepr<f32>, D>, mean: f32, std: f32) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    let norm = Normal::new(mean, std).unwrap();
    let mut gen = thread_rng();
    weight.mapv_inplace(|_| norm.sample(&mut gen));
    weight
}

//simply creates a weight and initializes each value to a number between 1 and -1
pub fn init_rand<D>(dim: Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    rand_weight(create_weight(dim), -1., 1.)
}

// xavier glorot's initialization function, used for tanh and sigmoid activations
// num_nodes : the number of input nodes (weights matrix.shape[1])]
// uses usize so you can directly pass .shape()[0] to it
pub fn init_xavier<D>(dim: Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    let weight = rand_weight(create_weight(dim.clone()), 0., 1.);
    let upper: f32 = 1. / (dim.clone()[0] as f32).sqrt();
    let lower: f32 = -1. / (dim[0] as f32).sqrt();
    scalar_mult(scalar_add(weight, lower), upper - lower)
}

// kaiming he's initialization function, used for ReLU activation
// num_nodes : the number of input nodes (weights matrix.shape[1])
pub fn init_he<D>(dim: Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D> where D: Dimension, {
    let weight = rand_weight(create_weight(dim.clone()), 0., 1.);
    let std: f32 = (2. / (dim[0] as f32)).sqrt();
    scalar_mult(rand_gaussian(weight, 0., 1.), std)
}

// calculates the exponential of a tensor
pub fn exp_weight<S,D>(mut weight: ArrayBase<S, D>) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    weight.mapv_inplace(|x: f32| x.exp());
    weight
}

// divides each value in a tensor by val
pub fn scalar_div<S,D>(mut weight: ArrayBase<S, D>, val: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    weight.mapv_inplace(|x: f32| x / val);
    weight
}

// multiplies each element in a tensor by val
pub fn scalar_mult<S,D>(mut weight: ArrayBase<S, D>, val: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    weight.mapv_inplace(|x: f32| x * val);
    weight
}

// subtracts val from each value in a 2D array
pub fn scalar_sub<S,D>(mut weight: ArrayBase<S, D>, val: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    weight.mapv_inplace(|x: f32| x - val);
    weight
}

// adds val to each value in an array
pub fn scalar_add<S,D>(mut weight: ArrayBase<S, D>, val: f32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    weight.mapv_inplace(|x: f32| x + val);
    weight
}

// replaces each value in a 2D array with its reciprocal (1/value)
pub fn scalar_reciprocal<S,D>(mut weight: ArrayBase<S, D>) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    weight.mapv_inplace(|x: f32| 1. / x);
    weight
}

// raises each value in a 2D array to the power of val
pub fn power_of<S,D>(mut weight: ArrayBase<S, D>, val: i32) -> ArrayBase<S, D> where S: DataMut<Elem = f32>, D: Dimension, {
    weight.mapv_inplace(|x: f32| x.powi(val));
    weight
}

// finds the largest element in an 2D-array and returns the index of it as a tuple
pub fn arg_max(weight: Array2<f32>) -> (usize, usize) {
    let mut max_indx: (usize, usize) = (0, 0);
    let mut max: f32 = weight[[0, 0]];
    let m = weight.shape();
    for i in 0..m[0] {
        for j in 0..m[1] {
            let v = weight[[i, j]];
            if v > max {
                max = v;
                max_indx = (i, j);
            }
        }
    }
    max_indx
}

pub fn transpose(weight: &Array2<f32>) -> Array2<f32> {
    let m = weight.shape();
    let mut out = create_weight(vec![m[1], m[0]]);

    for i in 0..m[0] {
        for j in 0..m[1] {
            out[[j,i]] = weight[[i,j]];
        }
    }
    out
}

pub fn flatten(weight: Array2<f32>) -> Array2<f32> {
    let m = weight.shape();
    let mut out = create_weight(vec![1,m[1] * m[0]]);
    let mut c = 0;
    for i in 0..m[0] {
        for j in 0..m[1] {
            out[[0,c]] = weight[[i,j]];
            c += 1;
        }
    }
    out
}

pub fn product(a: Array2<f32>, b: Array2<f32>) -> Array2<f32> {
    let ik: &[usize] = a.shape();
    let kj: &[usize] = b.shape();
    assert!(ik[1] == kj[0], "Incompatible shapes {:?} x {:?} : a.shape()[1] != b.shape()[0]", ik, kj);
    let mut outp: Array2<f32> = create_weight(vec![ik[0], kj[1]]);
    for i in 0..ik[0] {
        for j in 0..kj[1] {
            for k in 0..ik[1] {
                outp[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    outp
}
