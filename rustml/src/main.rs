use ndarray::Array2;

fn main() {
    //create the dimensions of each layer
    //planning to do mnist character recognition to build this around, so I will be using 28x28 images
    let input_layer = 28*28;
    let hidden_layer1 = 128;
    create_layer(1,3);
    //since there are 10 classes (0-9) the ai will print a "confidence matrix" that should sum to 1
    let output_layer = 10;
    let dim = vec![(input_layer,hidden_layer1),(hidden_layer1,output_layer)];
    //28*28x128x10 neural network is created
    //it is stored as [layer1_connections,layer2_ connections,...]
    let x = create_network(dim);
    println!("{:?}",x);
    println!("{}, {}",x.len(),x[1][[0,0]]);
}

/// dim : A list of the dimensions of the connections between each layer
/// network : A list of the connections between each layer as uninitialized 0's
pub fn create_network(dim : Vec<(usize,usize)>) -> Vec<Array2<f32>> {
    let mut network = Vec::new();
    for i in 0..dim.len() {
        let layer = create_layer(dim[i].0,dim[i].1);
        network.push(layer);
        println!("Creating layer {} with dimensions {},{}...",i,dim[i].0,dim[i].1)
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
    println!("{}",out);
    out
}

// calculates the exponential of a matrix
// out : a copy of the original matrix, where each value-x has been replaced by e^x
pub fn exponential(x : Vec<Array2<f32>>) -> Vec<Array2<f32>> {
    let mut out = x.to_vec();
    for l2d in out.iter_mut() {
        for l1d in l2d {
            *l1d = l1d.exp();
        }
    }
    out
}
