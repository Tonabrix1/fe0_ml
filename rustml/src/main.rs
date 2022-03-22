use ndarray::Array2;

fn main() {
    let dim = vec![(728,128),(128,10)];
    let x = create_network(dim);
    println!("{:?}",x)
}

fn create_network(dim : Vec<(usize,usize)>) -> Vec<Array2<f32>>{
    let mut network = Vec::new();
    for i in 0..dim.len() {
        let layer = create_layer(dim[i].0,dim[i].1);
        network.push(layer);
        println!("Creating layer {} with dimensions {},{}...",i,dim[i].0,dim[i].1)

     }
    return network;
}

fn create_layer(inner : usize, outer : usize) -> Array2<f32>{
    let out = Array2::<f32>::zeros((inner,outer));
    return out;
}
