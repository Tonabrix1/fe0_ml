use ndarray::Array2;
use rand::Rng;
use mnist_read;
use ndarray::s;

// Add structs to variable activation functions
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
    let mut my_nn = Net { weights : create_network(dim)};
    my_nn = init_rand(-1.,1.,my_nn);

    //let mut x = my_nn.weights[0].clone();
    //x = exp_layer(x);
    //println!("Randomized nn: {:?}\n\n\n\ne^weights[0]: {:?}",my_nn.weights[0], x);

    //let soft = softmax(x.clone());
    //should print a sum that is (within a reasonable margin of error) equal to 1
    //println!("Softmax: {:?}", soft.sum());

    //let sig = sigmoid(x);
    //should print a matrix of numbers between 0 and 1
    //println!("Sigmoid: {:?}", sig);

    //60k 28x28 images
    let train_x = load_image("train-images.idx3-ubyte",60000, 28*28);
    let train_x = train_x.slice(s![8i32..,..]);
    //some formatting issues
    let train_y = load_label("train-labels.idx1-ubyte",59992, 1);
    //10k 28x28 images
    let test_x = load_image("t10k-images.idx3-ubyte",10000,28*28);
    let test_x = test_x.slice(s![16i32..,..]);
    //more formatting issues
    let test_y = load_label("t10k-labels.idx1-ubyte",9992,1);

    let ndx : i32= 20;
    let slc = train_x.slice(s![ndx,..]);
    println!("Images: {:?}", train_x);
    println!("Image {}: {:?}",ndx,slc.clone());
    println!("Sum of pixels: {:?}",slc.sum());
    println!("{:?}", train_y);
}

fn train(mut net : Net, x : Array2<f64>, y : Array2<f64>, epochs : i32) {
    let lr : f32 = .001;
    let batch : i32 = 128;
    for epoch in 0..epochs {


        let mut targets = Array2::<f64>::zeros((y.clone().shape()[0],10));
        targets.slice_mut(s![..,10i32]);
        let forward_prop1 : Array2<f64> = activate_layer(x.clone(),net.weights[0].clone(), &sigmoid);
        let forward_prop2 : Array2<f64> = activate_layer(net.weights[0].clone(),net.weights[1].clone(), &softmax);

        let mut error : Array2<f64>= 2.*forward_prop2.clone()-targets / (forward_prop2.clone().shape()[0] as f64*derive_softmax(net.weights[1].clone()));
        let back_prop2 : Array2<f64> = forward_prop1.clone() * error.clone();

        error = (net.weights[1].clone().dot(&error.clone().t())).t().to_owned() * derive_sigmoid(x.clone().dot(&net.weights[0]));


        let back_prop1 : Array2<f64> = &x.t()*error.clone();
    }

}


// specifically for loading images from the mnist dataset
pub fn load_image(filename : &str, length1 : usize, length2 : usize) -> Array2<f64>{
    let tmp_u8 : Vec<u8>= mnist_read::read_data(&filename);
    let tmp_f64 : Vec<f64> = tmp_u8.into_iter().map(|d|d as f64 / 255.).collect();
    println!("Number of examples in {}: {}",filename,tmp_f64.len()/length2);
    let images : Array2<f64> = ndarray::Array::from_shape_vec((length1, length2), tmp_f64).expect("Bad data");
    images
}

// specifically for loading labels from the mnist dataset
pub fn load_label(filename : &str, length1 : usize, length2 : usize) -> Array2<usize>{
    let tmp_u8 : Vec<u8> = mnist_read::read_data(&filename);
    let usize_labels:Vec<usize> = tmp_u8.into_iter().map(|l|l as usize).collect();
    println!("Number of examples in {}: {}",filename,usize_labels.len()/length2);
    let array_labels : Array2<usize> = ndarray::Array::from_shape_vec((length1, length2), usize_labels).expect("Bad labels");
    array_labels
}

//pub fn sample_random()

//to-do find a way to store functions as objects so full forward prop can be done
pub fn activate_layer(x: Array2<f64>, y : Array2<f64>, activation : &dyn Fn(Array2<f64>) -> Array2<f64>) -> Array2<f64> {
    let out = activation(x.dot(&y));
    out
}

/// dim : A list of the dimensions of the connections between each layer
/// network : A list of the connections between each layer as uninitialized 0's
pub fn create_network(dim : Vec<(usize,usize)>) -> Vec<Array2<f64>> {
    let mut network = Vec::new();
    for i in 0..dim.len() {
        let layer = create_layer(dim[i].0,dim[i].1);
        network.push(layer);
        println!("Creating layer {} with dimensions {},{}...",i,dim[i].0,dim[i].1);
        //println!("{}",network[i]);
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

//subtracts val from each value in a 2D array
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
    let out = scalar_sub(layer,-1. * val);
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
    let out = 1./(scalar_add(exp_layer(-1. * layer),1.));
    out
}

//derivative of the sigmoid function
pub fn derive_sigmoid(layer : Array2<f64>) -> Array2<f64> {
    let ex = exp_layer(-1.*layer);
    let denom = scalar_add(ex.clone(),1.);
    //e^-x/((e^-1)+1)^2 = e^-x/((e^-1)+1)*((e^-1)+1)
    let out = ex/(denom.clone()*denom);
    out
}
