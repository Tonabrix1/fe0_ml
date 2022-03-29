use ndarray::Array2;
use rand::Rng;
use mnist_read;
use ndarray::s;
use rand::seq::SliceRandom;
//use std::io::Write;

// Add structs to variable activation functions
pub struct Net {
    pub weights : Vec<Array2<f32>>,
    pub biases : Vec<Array2<f32>>,
    pub activations : Vec<Box<dyn Fn(Array2<f32>) -> Array2<f32>>>,
}

pub struct Sample (Array2<f32>,Array2<f32>);

fn main() {
    // create the dimensions of each layer
    // planning to do mnist character recognition to build this around, so I will be using 28x28 images
    let input_layer = 28*28;
    let hidden_layer1 = 128;
    // since there are 10 classes (0-9) the ai will print a "confidence matrix"
    let output_layer = 10;
    let dim = vec![(input_layer, hidden_layer1),(hidden_layer1, output_layer)];
    // 28*28x128x10 neural network is created
    // it is stored as [layer1_connections,layer2_ connections,...]
    let mut my_nn = Net { weights : create_network(dim.clone()), biases : generate_bias(dim), activations : vec![Box::new(&ReLU), Box::new(&ReLU), Box::new(&softmax)]};
    my_nn = init_rand(-1.,1.,my_nn);

    //let mut x = my_nn.weights[0].clone();
    //x = exp_layer(my_nn.weights[0].clone());
    //println!("Randomized nn: {:?}\n\n\n\ne^weights[0]: {:?}",my_nn.weights[0], x);

    //let soft = softmax(my_nn.weights[0].clone());
    // should print a sum that is (within a reasonable margin of error) equal to 1
    //println!("Softmax: {:?}", soft.sum());

    //let sig = sigmoid(my_nn.weights[1].clone());
    // should print a matrix of numbers between 0 and 1
    //println!("Sigmoid: {:?}", sig);

    // 60k 28x28 images loaded as a single array
    let train_x = load_image("train-images.idx3-ubyte",60000, input_layer);
    // due to imperfect loading some of the images didn't get loaded
    let train_x = train_x.slice(s![8i32..,..]);
    // some formatting issues
    let train_y = load_label("train-labels.idx1-ubyte",59992, 1);
    // 10k images
    let test_x = load_image("t10k-images.idx3-ubyte",10000,input_layer);
    let test_x = test_x.slice(s![16i32..,..]);
    // more formatting issues
    let test_y = load_label("t10k-labels.idx1-ubyte",9992,1);

    let ndx : i32 = 20;
    let slc = train_x.slice(s![ndx,..]);
    println!("Images: {:?}", train_x);
    println!("Image {}: {:?}",ndx,slc.clone());
    println!("Sum of pixels: {:?}",slc.sum());
    println!("{:?}", train_y);

    // this is a terrible way of doing it I'm aware :hmm:
    let dim_train_x = train_x.slice(s![..,0i32]).len();

    let samples = sample(dim_train_x as i32);
    //let smp = ndarray::Array::from(samples.clone());
    let mut dataset = generate_dataset(train_x.to_owned(),train_y, samples,10);

    train(my_nn,dataset, 1000, None, None);

    //saves an image to be opened by validator.py which can be used to validate that the labels are correctly matched to the images
    //let mut file = std::fs::File::create("data.txt").expect("create failed");
    //sample.0.map(|d| file.write_all((d.to_string() + " ").as_bytes()));

}

pub fn train(mut net : Net, mut dataset : Vec<Sample>, epochs : i32, batch : Option<i32>, lr : Option<f32>){
    let lr : f32 = lr.unwrap_or(0.001);
    let batch : i32 = batch.unwrap_or(128);

    // second tuple element of generate_dataset
    let mut targets : Array2<f32>;
    let mut guess_onehot : Array2<f32>;
    let mut accuracies : Vec<f32> = Vec::new();
    let mut losses : Vec<f32> = Vec::new();
    let mut next_weights : Vec<Array2<f32>>;
    let mut next_biases : Vec<Array2<f32>>;

    for epoch in 0..epochs {
        next_weights = Vec::new();
        next_biases = Vec::new();
        targets = ndarray::Array2::<f32>::zeros((1,10));
        guess_onehot = targets.clone();
        println!("Starting epoch {}",epoch);
        // new sample
        let sample : Sample = dataset.pop().expect("Not enough samples");
        let x : Array2<f32> = sample.0.clone();
        let y : Array2<f32> = sample.1;

        let (fp_acts, fp_dots, error) = forward_propagate(net, x, y);
        let out = fp_acts.last().expect("empty activations vector").clone();


        let category = arg_max(out.clone());
        println!("argmax: {:?}", category);
        guess_onehot[[0,category.1.clone()]] = 1.;
        let accuracy = y.clone()[[category.clone().1,0]];
        accuracies.push(accuracy);
        //println!("guess onehot: {:?}", guess_onehot.clone());
        //println!("y onehot: {:?}", y.clone());
        let loss = mean_squared_error(guess_onehot.clone(),y.clone().t().to_owned()).expect("MSE failed");
        losses.push(loss);
        println!("Raw AI guess: {:?}", forward_prop2);
        println!("AI guess: {}",category.1);
        println!("Real answer: {}",arg_max(y.clone()).0);

        println!("weights shape : {:?}, back_prop1 shape: {:?}", net.weights[0].clone().shape(),back_prop1.clone().shape());
        next_weights.push(net.weights[0].clone()-lr*back_prop1.clone().t().to_owned());
        println!("b1 shape: {:?}, back_prop1 shape: {:?}", net.biases[0].clone().shape(), back_prop1.clone().t().to_owned().shape());
        next_biases.push(net.biases[0].clone()-lr*back_prop2.t().to_owned());
        next_weights.push(net.weights[1].clone()-lr*back_prop2.clone().t().to_owned());
        println!("b2 shape: {:?}, back_prop2 shape: {:?}", net.biases[1].clone().shape(), back_prop2.clone().t().to_owned().shape());
        next_biases.push(net.biases[1].clone()-lr*back_prop1.t().to_owned());
        println!("Epoch #{}, loss: {}", epoch, loss);
        net.weights = next_weights;
        net.biases = next_biases;
        //println!("{:?}", net.biases[0].clone());
    }

}


// returns activated weights, dots, error
pub fn forward_propagate(net : Net, x : Array2<f32>, y : Array2<f32>) -> (Vec<Array2<f32>>,Vec<Array2<f32>>, Array2<f32>){
    // initializing variables
    let mut activated : Vec<Array2<f32>> = Vec::new();
    let mut dots : Vec<Array2<f32>> = Vec::new();
    let mut curr_a;
    let mut curr_b;
    // could be cleaner if I inserted x into the first weights vector but this is simpler overall
    for i in 0..net.weights.len() {
        curr_b = net.weights[i].clone();
        if i <= 0 { curr_a = x.clone(); }
        else { curr_a = activated[i-1].clone(); }
        assert_eq!(curr_a.clone().shape()[1],curr_b.clone().shape()[0]);
        let (act, dot) = activate_layer(curr_a.clone(), curr_b.clone(),  net.biases[i].clone(), &*net.activations[i]);
        activated.push(act);
        dots.push(dot);
    }
    let error = activated.last().expect("activated is empty").clone() - y.clone();
    (activated, dots, error)
}


// preforms back propagation
pub fn back_propagate(mut net : Net, y : Array2<f32>, error : Array2<f32>, batch_size : Option<i32>, lr : Option<f32>, lr_gamma : Option<f32>) {
    // unwrapping optional arguements
    let batch_size = batch_size.unwrap_or(128);
    // learning rate, multiplied by true weight updates
    let lr = lr.unwrap_or(0.001);
    // learning rate, multiplied by true bias updates
    let lr_gamma = lr_gamma.unwrap_or(*&lr);
    // derivative of cost
    let d_cost = (1./batch_size as f32)*error;
    // initializing variables
    let mut d_weights : Vec<Array2<f32>> = Vec::new();
    let mut d_biases : Vec<Array2<f32>> = Vec::new();
    let mut curr_dw;
    let mut curr_db;
    for i in 0..net.weights.len() {
        if i <= 0 { curr_dw = ; }
    }
}



// creates a stack (len batch_size) of random integers from range 0..dim
// used to select an image and label pair for each training function
pub fn sample(num : i32) -> Vec<i32>{
    let mut stack : Vec<i32> = (0..num).collect();
    stack.shuffle(&mut rand::thread_rng());
    stack
}


// takes a stack (Vec<i32>) of stamples from the sample method and creates a stack of (image,label) tuples
pub fn generate_dataset(x : Array2<f32>, y : Array2<usize>, mut samples : Vec<i32>, output_layers : usize) -> Vec<Sample> {
    println!("Generating dataset...");
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

//generates a list of biases with the same dimensions as the dot products of
pub fn generate_bias(dim : Vec<(usize, usize)>) -> Vec<Array2<f32>>{
    println!("Generating bias...");
    let mut out = Vec::new();
    for i in 0..dim.len() {
        let mut bias = create_layer(1,dim[i].1);
        bias = rand_layer(0.,1.,bias);
        //println!("bias: {:?}", bias);
        out.push(bias);
    }
    out
}

// creates a random ly initialized matrix with the dimensions of layer2.dot(layer1)
pub fn generate_bias_layer(layer1 : Array2<f32>, layer2 : Array2<f32>, min : Option<f32>, max : Option<f32>) -> Array2<f32> {
    let temp_layer = layer2.dot(&layer1);
    rand_layer(min.unwrap_or(0f32), max.unwrap_or(1f32), temp_layer)
}

// specifically for loading images from the mnist dataset
pub fn load_image(filename : &str, length1 : usize, length2 : usize) -> Array2<f32>{
    let tmp_u8 : Vec<u8> = mnist_read::read_data(&filename);
    //normalize the values from 0-255 to 0-1
    let tmp_f32 : Vec<f32> = tmp_u8.into_iter().map(|d|d as f32 / 255.).collect();
    println!("Number of examples in {}: {}",filename,tmp_f32.len()/length2);
    let images : Array2<f32> = ndarray::Array::from_shape_vec((length1, length2), tmp_f32).expect("Bad images");
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

//to-do find a way to store functions as objects so full forward prop can be done from a single object
pub fn activate_layer(weight : Array2<f32>, prev : Array2<f32>, bias : Array2<f32>, activation : &dyn Fn(Array2<f32>) -> Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    println!("activating shapes {:?}, {:?}", weight.clone().shape(),prev.clone().shape());
    let dt = weight.dot(&prev);
    println!("dt shape:  {:?}, bias shape: {:?}", dt.shape(), bias.shape());
    let dt_b = dt + bias;
    println!("activating...");
    let out = activation(dt_b.clone());
    (out, dt_b)
}

/// dim : A list of the dimensions of the connectiions between each layer
/// network : A list of the connections between each layer as uninitialized 0's
pub fn create_network(dim : Vec<(usize,usize)>) -> Vec<Array2<f32>> {
    println!("Creating network...");
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
#[allow(non_snake_case)]
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


// finds the largest element in an 2D-array and returns the index of it
pub fn arg_max(arr : Array2<f32>) -> (usize,usize) {
    let mut max_indx : (usize,usize) = (0, 0);
    let mut max : f32 = arr[[0, 0]];
    for i in 0..arr.shape()[0]{
        for j in  0..arr.shape()[1] {
            if arr[[i,j]] > max {
                max = arr[[i,j]];
                max_indx = (i,j);
            }
        }
    }
    max_indx
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
            out[[i,j]] -= val;
        }
    }
    out
}

// calls scalar_sub and multiplies val by -1.0
pub fn scalar_add(layer: Array2<f32>, val : f32) -> Array2<f32>{
    let out = scalar_sub(layer,-1. * val);
    out
}

pub fn power_of(arr : Array2<f32>, val : i32) -> Array2<f32>{
    arr.mapv(|x| x.powi(val))
}

pub fn mat_mul(arr1 : Array2<f32>, arr2 : Array2<f32>) -> Array2<f32>{
    let m1 = arr1.clone().shape()[0];
    let m2 = arr2.clone().shape()[1];
    let mut outp : Array2<f32> = create_layer(m1,m2);
    for i in 0..m1 {
        for j in 0..m2 {
            for k in 0..arr2.clone().shape()[0] {
                outp[[i, j]] = outp[[i, j]] + (arr1[[i,k]] * arr2[[k,j]]);
            }
        }
    }
    outp
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

pub fn mean_squared_error(x : Array2<f32>, y : Array2<f32>) -> Option<f32>{
    let loss = power_of(x-y,2).mean();
    loss
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

#[allow(non_snake_case)]
pub fn ReLU(mut x : Array2<f32>) -> Array2<f32>{
    for i in 0..x.clone().shape()[0] {
        for j in 0..x.clone().shape()[1] {
            if x[[i,j]] >= 1. { x[[i,j]] = 1.; } else { x[[i,j]] = 0.; }
        }
    }
    x
}

#[allow(non_snake_case)]
pub fn derive_ReLU(mut x : Array2<f32>) -> Array2<f32> {
    // technically it's undefined at x[[i,j]] == 0
    for i in 0..x.clone().shape()[0] {
        for j in 0..x.clone().shape()[1] {
            if x[[i,j]] > 0. { x[[i,j]] = 1.; } else { x[[i,j]] = 0.; }
        }
    }
    x
}
