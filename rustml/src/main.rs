#[allow(non_snake_case)]
use ndarray::Array2;
use rand::Rng;
use mnist_read;
use ndarray::s;
use rand::seq::SliceRandom;

// Add structs to variable activation functions
pub struct Net {
    pub weights : Vec<Array2<f32>>,
}

pub struct Sample (Array2<f32>,Array2<f32>);

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

    let mut x = my_nn.weights[0].clone();
    x = exp_layer(my_nn.weights[0].clone());
    println!("Randomized nn: {:?}\n\n\n\ne^weights[0]: {:?}",my_nn.weights[0], x);

    let soft = softmax(my_nn.weights[0].clone());
    //should print a sum that is (within a reasonable margin of error) equal to 1
    println!("Softmax: {:?}", soft.sum());

    let sig = sigmoid(my_nn.weights[0].clone());
    //should print a matrix of numbers between 0 and 1
    println!("Sigmoid: {:?}", sig);
    //60k 28x28 images loaded as a single array
    let train_x = load_image("train-images.idx3-ubyte",60000, input_layer);
    //due to imperfect loading some of the images didn't get loaded
    let train_x = train_x.slice(s![8i32..,..]);
    //some formatting issues
    let train_y = load_label("train-labels.idx1-ubyte",59992, 1);
    //10k images
    let test_x = load_image("t10k-images.idx3-ubyte",10000,input_layer);
    let test_x = test_x.slice(s![16i32..,..]);
    //more formatting issues
    let test_y = load_label("t10k-labels.idx1-ubyte",9992,1);

    let ndx : i32= 20;
    let slc = train_x.slice(s![ndx,..]);
    println!("Images: {:?}", train_x);
    println!("Image {}: {:?}",ndx,slc.clone());
    println!("Sum of pixels: {:?}",slc.sum());
    println!("{:?}", train_y);

    //this is a terrible way of doing it I'm aware :hmm:
    let dim_train_x = train_x.slice(s![..,0i32]).len();

    let samples = sample(dim_train_x as i32);
    let smp = ndarray::Array::from(samples.clone());
    let mut dataset = generate_dataset(train_x.to_owned(),train_y, samples,10);
    let sample = dataset.pop();
    println!("Random choices out of {}, {:?}",dim_train_x ,smp);
    println!("random sample: {:?}", sample.unwrap().0);

}

fn train(mut net : Net, x : Array2<f32>, y : usize, epochs : i32, batch : Option<i32>, lr : Option<f32>) {
    let lr : f32 = lr.unwrap_or(0.001);
    let batch : i32 = batch.unwrap_or(128);

    //second tuple element of generate_dataset
    let targets = ndarray::Array2::<f32>::zeros((1,10));

    for epoch in 0..epochs {
        // activate(hidden_layer1.dot(inputs))
        let forward_prop1 : Array2<f32> = activate_layer(net.weights[0].clone(),x.clone(), &sigmoid);
        // activate(hidden_layer2.dot(hidden_layer1))
        let forward_prop2 : Array2<f32> = activate_layer(net.weights[1].clone(),net.weights[0].clone(), &softmax);

        // 2 * (output - label) /  (output.shape[0] * derive_softmax(hidden_layer2))
        let mut error : Array2<f32> = 2. * forward_prop2.clone() - targets.clone() / (forward_prop2.clone().shape()[0] as f32*derive_softmax(net.weights[1].clone()));
        let back_prop2 : Array2<f32> = forward_prop1.clone() * error.clone();

        error = (net.weights[1].clone().dot(&error.clone().t())).t().to_owned() * derive_sigmoid(x.clone().dot(&net.weights[0]));


        let back_prop1 : Array2<f32> = &x.t()*error.clone();

    }

}


//creates a stack (len batch_size) of random integers from range 0..dim
//used to select an image and label pair for each training function
fn sample(num : i32) -> Vec<i32>{
    let mut stack : Vec<i32> = (0..num).collect();
    stack.shuffle(&mut rand::thread_rng());
    stack
}


// takes a stack (Vec<i32>) of stamples from the sample method and creates a stack of (image,label) tuples
fn generate_dataset(x : Array2<f32>, y : Array2<usize>, mut samples : Vec<i32>, output_layers : usize) -> Vec<Sample> {
    let mut outp : Vec<Sample> = Vec::new();
    while let Some(sample) = samples.pop() {
        let image = x.slice(s![sample..sample+1,..]).to_owned();
        let mut one_hot = Array2::<f32>::zeros((output_layers,1));
        let label = y[[sample as usize,0]];
        one_hot[[label,0]] = 1.;
        let curr : Sample = Sample(image, one_hot);
        outp.push(curr);
    }
    outp
}

//creates a random ly initialized matrix with the dimensions of layer2.dot(layer1)
fn generate_bias(layer1 : Array2<f32>, layer2 : Array2<f32>, min : Option<f32>, max : Option<f32>) -> Array2<f32> {
    let temp_layer = layer2.dot(&layer1);
    rand_layer(min.unwrap_or(0f32), max.unwrap_or(1f32), temp_layer)
}

// specifically for loading images from the mnist dataset
pub fn load_image(filename : &str, length1 : usize, length2 : usize) -> Array2<f32>{
    let tmp_u8 : Vec<u8> = mnist_read::read_data(&filename);
    //normalize the values from 0-255 to 0-1
    let tmp_f32 : Vec<f32> = tmp_u8.into_iter().map(|d|d as f32 / 255.).collect();
    println!("Number of examples in {}: {}",filename,tmp_f32.len()/length2);
    let images : Array2<f32> = ndarray::Array::from_shape_vec((length1, length2), tmp_f32).expect("Bad data");
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

//to-do find a way to store functions as objects so full forward prop can be done
pub fn activate_layer(weight : Array2<f32>, prev : Array2<f32>, activation : &dyn Fn(Array2<f32>) -> Array2<f32>) -> Array2<f32> {
    let out = activation(weight.dot(&prev));
    out
}

/// dim : A list of the dimensions of the connectiions between each layer
/// network : A list of the connections between each layer as uninitialized 0's
pub fn create_network(dim : Vec<(usize,usize)>) -> Vec<Array2<f32>> {
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
pub fn rand_layer(x : f32,y : f32, layer : Array2<f32>) -> Array2<f32> {
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
pub fn exp_layer(layer : Array2<f32>) -> Array2<f32> {
    let mut layer = layer.clone();
    for i in 0..layer.shape()[0] {
        for j in 0..layer.shape()[1] {
            layer[[i,j]] = layer[[i,j]].exp(); // e^layer[[i,j]]
        }
    }
    layer
}

//subtracts val from each value in a 2D array
pub fn scalar_sub(layer : Array2<f32>, val : f32) -> Array2<f32> {
    let mut out = layer.clone();
    for i in 0..out.shape()[0] {
        for j in 0..out.shape()[1] {
            out[[i,j]] = out[[i,j]] - val;
        }
    }
    out
}

// calls scalar_sub and multiplies val by -1.0
pub fn scalar_add(layer: Array2<f32>, val : f32) -> Array2<f32>{
    let out = scalar_sub(layer,-1. * val);
    out
}

// creates a matrix of probabilities summing to 1 (or at least close enough :^)
// layer : the original 2D matrix
// out : the output of the softmax function (at least an algebraically equivalent function)
pub fn softmax(layer : Array2<f32>) -> Array2<f32> {
    let ex = exp_layer(layer);
    let out = ex.clone()/ex.sum();
    out
}

// the derivative of the softmax function
pub fn derive_softmax(layer : Array2<f32>) -> Array2<f32> {
    let sf = softmax(layer);
    sf.clone() * scalar_sub(sf,1.)
}

// take each value in an array and scales it into a number between 0 and 1
pub fn sigmoid(layer : Array2<f32>) -> Array2<f32> {
    // 1/(e^-x)+1
    let out = 1./(scalar_add(exp_layer(-1. * layer),1.));
    out
}

//derivative of the sigmoid function
pub fn derive_sigmoid(layer : Array2<f32>) -> Array2<f32> {
    //e^-x
    let ex = exp_layer(-1.*layer);
    //(e^-x)+1
    let denom = scalar_add(ex.clone(),1.);
    //e^-x/((e^-1)+1)^2 = e^-x/((e^-1)+1)*((e^-1)+1)
    let out = ex/(denom.clone()*denom);
    out
}
