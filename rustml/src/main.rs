use ndarray::Array2;
use rand::Rng;

//for now just using this as shorthand
pub struct Net {
    pub weights : Vec<Array2<f64>>,
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

    let soft = softmax(x.clone());
    //should print a sum that is (within a reasonable margin of error) equal to 1
    println!("Softmax: {:?}", soft.sum());

    let sig = sigmoid(x);
    //should print a matrix of numbers between 0 and 1
    println!("Sigmoid: {:?}", sig);

}

/// dim : A list of the dimensions of the connections between each layer
/// network : A list of the connections between each layer as uninitialized 0's
pub fn create_network(dim : Vec<(usize,usize)>) -> Vec<Array2<f64>> {
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
pub fn create_layer(len_2D : usize, len_1D : usize) -> Array2<f64> {
    //initializes an array of 0's: len_2D=1; len_1D=3
    // [[0, 0, 0]]
    //the first element is accessed with Array2[[0,0]]
    let out = Array2::<f64>::zeros((len_2D,len_1D));
    out
}

// randomizes the values of weights in a Net object
// net : A Net object where all weights have been passed through rand_layer(x,y,weight)
pub fn init_rand(x : f64, y : f64, mut net : Net) -> Net{
    for i in 0..net.weights.len() {
        net.weights[i] = rand_layer(x,y,net.weights[i].clone());
    }
    net
}

// replaces all values in a 2D matrix with a random number between x and y (inclusive)
// layer : the original 2D matrix
// out : the transformed 2D matrix where all values have been randomized
pub fn rand_layer(x : f64,y : f64, layer : Array2<f64>) -> Array2<f64> {
    let mut out = layer.clone();
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
pub fn exp_layer(layer : Array2<f64>) -> Array2<f64> {
    let mut layer = layer.clone();
    for i in 0..layer.shape()[0] {
        for j in 0..layer.shape()[1] {
            layer[[i,j]] = layer[[i,j]].exp(); // e^layer[[i,j]]
        }
    }
    layer
}

//creates an array with the same dimensions of layer where all the values are equal to val and subtracts it from layer
pub fn scalar_sub(layer : Array2<f64>, val : f64) -> Array2<f64> {
    let mut out = layer.clone();
    for i in 0..out.shape()[0] {
        for j in 0..out.shape()[1] {
            out[[i,j]] = out[[i,j]] - val;
        }
    }
    out
}

//calls scalar_sub and multiplies val by -1.0
pub fn scalar_add(layer: Array2<f64>, val : f64) -> Array2<f64>{
    let out = scalar_sub(layer,-1.*val);
    out
}

// creates a matrix of probabilities summing to 1 (or at least close enough :^)
// layer : the original 2D matrix
// out : the output of the softmax function (at least an algebraically equivalent function)
pub fn softmax(layer : Array2<f64>) -> Array2<f64> {
    let ex = exp_layer(layer);
    let out = ex.clone()/ex.sum();
    out
}

// the derivative of the softmax function
pub fn derive_softmax(layer : Array2<f64>) -> Array2<f64> {
    let sf = softmax(layer);
    return sf.clone() * scalar_sub(sf,1.);
}

// take each value in an array and scales it into a number between 0 and 1
pub fn sigmoid(layer : Array2<f64>) -> Array2<f64> {
    let out = 1./(scalar_add(exp_layer(-1.*layer),1.));
    out
}

//derivative of the sigmoid function
pub fn derivative_sigmoid(layer : Array2<f64>) -> Array2<f64> {
    let ex = exp_layer(-1.*layer);
    let denom = scalar_add(ex.clone(),1.);
    //e^-x/((e^-1)+1)^2
    let out = ex/(denom.clone()*denom);
    out
}
