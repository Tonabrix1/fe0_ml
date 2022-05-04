use ndarray::s;
mod FE0_ML;

fn main() {
    // create the dimensions of each layer
    // planning to do mnist character recognition to build this around, so I will be using 28x28 images
    let input_layer = 28 * 28;
    let hidden_layer1 = 128;
    // since there are 10 classes (0-9) the ai will print a "confidence matrix"
    let output_layer = 10;
    let dim = vec![(input_layer, hidden_layer1), (hidden_layer1, output_layer)];
    // 28*28x128x10 neural network is created
    // it is stored as [layer1_connections,layer2_ connections,...]
    let mut my_nn = FE0_ML::Net {
        weights: FE0_ML::create_network(dim.clone()),
        biases: FE0_ML::generate_bias(dim),
        activations: vec![Box::new(&FE0_ML::ReLU), Box::new(&FE0_ML::ReLU), Box::new(&FE0_ML::softmax)],
        derivative_activations: vec![
            Box::new(&FE0_ML::derive_ReLU),
            Box::new(&FE0_ML::derive_ReLU),
            Box::new(&FE0_ML::derive_softmax),
        ],
    };
    my_nn = FE0_ML::init_rand(-1., 1., my_nn);

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
    let train_x = FE0_ML::load_image("train-images.idx3-ubyte", 60000, input_layer);
    // due to imperfect loading some of the images didn't get loaded
    let train_x = train_x.slice(s![8i32.., ..]);
    // some formatting issues
    let train_y = FE0_ML::load_label("train-labels.idx1-ubyte", 59992, 1);
    // 10k images
    let test_x = FE0_ML::load_image("t10k-images.idx3-ubyte", 10000, input_layer);
    let test_x = test_x.slice(s![16i32.., ..]);
    // more formatting issues
    let test_y = FE0_ML::load_label("t10k-labels.idx1-ubyte", 9992, 1);

    // this is a terrible way of doing it I'm aware :hmm:
    let dim_train_x = train_x.slice(s![.., 0i32]).len();

    let samples = FE0_ML::sample(dim_train_x as i32);
    //let smp = ndarray::Array::from(samples.clone());
    let mut dataset = FE0_ML::generate_dataset(train_x.to_owned(), train_y, samples, 10);

    FE0_ML::train(my_nn, dataset, 1000, None, None);

    //saves an image to be opened by validator.py which can be used to validate that the labels are correctly matched to the images
    //let mut file = std::fs::File::create("data.txt").expect("create failed");
    //sample.0.map(|d| file.write_all((d.to_string() + " ").as_bytes()));
}

