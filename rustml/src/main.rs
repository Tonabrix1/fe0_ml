use ndarray::Array2;

fn main() {
    //create the dimensions of each layer
    //planning to do mnist character recognition to build this around, so I will be using 28x28 images
    let input_layer = 28*28;
    let hidden_layer1 = 128;
    //since there are 10 classes (0-9) the ai will print a "confidence matrix" that should sum to 1
    let output_layer = 10;
    //28*28x128x10 neural network is created
    let dim = vec![(input_layer,hidden_layer1),(hidden_layer1,output_layer)];
    let x = create_network(dim);
    println!("{:?}",x)
}

/// dim : A list of the dimensions of the connections between each layer
/// network : A list of the connections between each layer as uninitialized
fn create_network(dim : Vec<(usize,usize)>) -> Vec<Array2<f32>>{
    let mut network = Vec::new();
    for i in 0..dim.len() {
        let layer = create_layer(dim[i].0,dim[i].1);
        network.push(layer);
        println!("Creating layer {} with dimensions {},{}...",i,dim[i].0,dim[i].1)
     }
    return network;
}

// inner : the dimension of input matrix-a
// outer : the dimensions of input matrix-b
// out : a 2D array (ndarray::Array2) of the connections between matrix-a and matrix-b
fn create_layer(inner : usize, outer : usize) -> Array2<f32> {
    let out = Array2::<f32>::zeros((inner,outer));
    return out;
}
