use ndarray::Array2;
use rand::Rng;

//for now just using this as shorthand
pub struct Net {
    pub weights : Vec<Array2<f32>>,
}

fn main() {
    //create the dimensions of each layer
    //planning to do mnist character recognition to build this around, so I will be using 28x28 images
    let input_layer = 28*28;
    let hidden_layer1 = 128;
    //since there are 10 classes (0-9) the ai will print a "confidence matrix"
    let output_layer = 10;
    let dim = vec![(input_layer,hidden_layer1),(hidden_layer1,output_layer)];
    //28*28x128x10 neural network is created
    //it is stored as [layer1_connections,layer2_ connections,...]
    let mut my_nn = Net{weights:create_network(dim)};
    my_nn = init_rand(-1.,1.,my_nn);

    let mut x = my_nn.weights[0].clone();
    x = exp_layer(x);
    println!("Randomized nn: {:?}\n\n\n\ne^weights[0]: {:?}",my_nn.weights[0], x);

    let

}

/// dim : A list of the dimensions of the connections between each layer
/// network : A list of the connections between each layer as uninitialized 0's
pub fn create_network(dim : Vec<(usize,usize)>) -> Vec<Array2<f32>> {
    let mut network = Vec::new();
    for i in 0..dim.len() {
        let layer = create_layer(dim[i].0,dim[i].1);
        network.push(layer);
        println!("Creating layer {} with dimensions {},{}...",i,dim[i].0,dim[i].1);
        println!("{}",network[i]);
     }
    network
}

// len2D : layer.len() == len2D
// len1D : layer[n].len() == len1D
// out : a 2D array (ndarray::Array2) of the connections between matrix-a and matrix-b
pub fn create_layer(len_2D : usize, len_1D : usize) -> Array2<f32> {
    //initializes an array of 0's: len_2D=1; len_1D=3
    // [[0, 0, 0]]
    //the first element is accessed with Array2[[0,0]]
    let out = Array2::<f32>::zeros((len_2D,len_1D));
    out
}

// randomizes the values of weights in a Net object
// net : A Net object where all weights have been passed through rand_layer(x,y,weight)
pub fn init_rand(x : f32, y : f32, mut net : Net) -> Net{
    for i in 0..net.weights.len() {
        net.weights[i] = rand_layer(x,y,net.weights[i].clone());
    }
    net
}

// replaces all values in a 2D matrix with a random number between x and y (inclusive)
// layer : the original 2D matrix
// out : the transformed 2D matrix where all values have been randomized
pub fn rand_layer(x : f32,y : f32, mut layer : Array2<f32>) -> Array2<f32> {
    let mut out = layer;
    for i in 0..out.shape()[0] {
        for j in 0..out.shape()[1] {
            out[[i,j]] = rand::thread_rng().gen_range(x..y);
        }
    }
    out
}

// calculates the exponential of a matrix
// layer : the original 2D matrix
// out : the transformed 2D matrix where each value-x has been replaced by e^x
pub fn exp_layer(mut layer : Array2<f32>) -> Array2<f32> {
    for i in 0..layer.shape()[0] {
        for j in 0..layer.shape()[1] {
            layer[[i,j]] = layer[[i,j]].exp(); // e^layer[[i,j]]
        }
    }
    layer
}

// first cost function function [implement more of these ...]
// layer : the original 2D matrix
// out : the output of the softmax function (at least an algebraically equivalent function)
pub fn softmax(layer : Array2<f32>) -> Array2<f32> {
    let ex = exp_layer(layer);
    let out = ex.clone()/ex.sum();
    out
}

pub fn sigmoid(x : f32) {

}
