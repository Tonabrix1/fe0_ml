#![allow(dead_code, unused_variables)]

use ndarray::{Array, Array2, ArrayBase, ArrayView, DataMut, Dimension, OwnedRepr};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal};

//initializes a tensor of rank dim.len() where every element is a 0
pub fn create_weight<D>(dim: &Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D>
where
    D: Dimension,
{
    let mut shape = D::zeros(dim.len());
    shape.as_array_view_mut().assign(&ArrayView::from(dim));
    Array::zeros(shape)
}

// replaces all values in an ND matrix with a random number between x and y (inclusive)
// weight : the original ND matrix
// out : the transformed ND matrix where all values have been randomized
pub fn rand_weight<S, D>(weight: &mut ArrayBase<S, D>, x: f32, y: f32) -> &mut ArrayBase<S, D>
where
    S: DataMut<Elem = f32>,
    D: Dimension,
{
    weight.mapv_inplace(|_| thread_rng().gen_range(x..=y));
    weight
}

// replaces all values in an ND matrix with a random number taken from a gaussian (normal) distribution
pub fn rand_gaussian<D>(
    weight: &mut ArrayBase<OwnedRepr<f32>, D>,
    mean: f32,
    std: f32,
) -> &mut ArrayBase<OwnedRepr<f32>, D>
where
    D: Dimension,
{
    let norm = Normal::new(mean, std).unwrap();
    let mut gen = thread_rng();
    weight.mapv_inplace(|_| norm.sample(&mut gen));
    weight
}

//simply creates a weight and initializes each value to a number between 1 and -1
pub fn init_rand<D>(dim: &Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D>
where
    D: Dimension,
{
    rand_weight(&mut create_weight(dim), -1., 1.).to_owned()
}

// xavier glorot's initialization function, used for tanh and sigmoid activations
// num_nodes : the number of input nodes (weights matrix.shape[1])]
// uses usize so you can directly pass .shape()[0] to it
pub fn init_xavier<D>(dim: &Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D>
where
    D: Dimension,
{
    let empty: &mut ArrayBase<OwnedRepr<f32>, D> = &mut create_weight(dim);
    let weight = rand_weight(empty, 0., 1.);
    let upper: f32 = 1. / (dim.clone()[0] as f32).sqrt();
    let lower: f32 = -1. / (dim[0] as f32).sqrt();
    scalar_mult(scalar_add(weight, lower), upper - lower).to_owned()
}

// kaiming he's initialization function, used for ReLU activation
// num_nodes : the number of input nodes (weights matrix.shape[1])
pub fn init_he<D>(dim: &Vec<usize>) -> ArrayBase<OwnedRepr<f32>, D>
where
    D: Dimension,
{
    let empty: &mut ArrayBase<OwnedRepr<f32>, D> = &mut create_weight(dim);
    let weight = rand_weight(empty, 0., 1.);
    let std: f32 = (2. / (dim[0] as f32)).sqrt();
    scalar_mult(rand_gaussian(weight, 0., 1.), std).to_owned()
}

// calculates the exponential of a tensor
pub fn exp_weight<S, D>(weight: &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D>
where
    S: DataMut<Elem = f32>,
    D: Dimension,
{
    weight.mapv_inplace(|x: f32| x.exp());
    weight
}

// divides each value in a tensor by val
pub fn scalar_div<S, D>(weight: &mut ArrayBase<S, D>, val: f32) -> &mut ArrayBase<S, D>
where
    S: DataMut<Elem = f32>,
    D: Dimension,
{
    weight.mapv_inplace(|x: f32| x / val);
    weight
}

// multiplies each element in a tensor by val
pub fn scalar_mult<S, D>(weight: &mut ArrayBase<S, D>, val: f32) -> &mut ArrayBase<S, D>
where
    S: DataMut<Elem = f32>,
    D: Dimension,
{
    weight.mapv_inplace(|x: f32| x * val);
    weight
}

// subtracts val from each value in a 2D array
pub fn scalar_sub<S, D>(weight: &mut ArrayBase<S, D>, val: f32) -> &mut ArrayBase<S, D>
where
    S: DataMut<Elem = f32>,
    D: Dimension,
{
    weight.mapv_inplace(|x: f32| x - val);
    weight
}

// adds val to each value in an array
pub fn scalar_add<S, D>(weight: &mut ArrayBase<S, D>, val: f32) -> &mut ArrayBase<S, D>
where
    S: DataMut<Elem = f32>,
    D: Dimension,
{
    weight.mapv_inplace(|x: f32| x + val);
    weight
}

// replaces each value in a 2D array with its reciprocal (1/value)
pub fn scalar_reciprocal<S, D>(weight: &mut ArrayBase<S, D>) -> &mut ArrayBase<S, D>
where
    S: DataMut<Elem = f32>,
    D: Dimension,
{
    weight.mapv_inplace(|x: f32| 1. / x);
    weight
}

// raises each value in a 2D array to the power of val
pub fn power_of<S, D>(weight: &mut ArrayBase<S, D>, val: i32) -> &mut ArrayBase<S, D>
where
    S: DataMut<Elem = f32>,
    D: Dimension,
{
    weight.mapv_inplace(|x: f32| x.powi(val));
    weight
}

// finds the largest element in an 2D-array and returns the index of it as a tuple
pub fn arg_max(weight: &Array2<f32>) -> (usize, usize) {
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
    let mut out = create_weight(&vec![m[1], m[0]]);

    for i in 0..m[0] {
        for j in 0..m[1] {
            out[[j, i]] = weight[[i, j]];
        }
    }
    out
}

pub fn flatten(weight: &Array2<f32>) -> Array2<f32> {
    let m = weight.shape();
    let mut out = create_weight(&vec![1, m[1] * m[0]]);
    let mut c = 0;
    for i in 0..m[0] {
        for j in 0..m[1] {
            out[[0, c]] = weight[[i, j]];
            c += 1;
        }
    }
    out
}
