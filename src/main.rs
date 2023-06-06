mod activations;
mod layers;
mod matrixutil;
mod netutil;
mod cost;
use crate::activations::{Activations};
use crate::layers::Layers::{Dense};
use crate::netutil::{Sequential, Net, Sample};
use crate::matrixutil::{init_rand, create_weight, transpose, flatten};
use crate::cost::{Cost};
use ndarray::{s, Array3, Array2, Ix2};
use mnist::*;

fn main() {
    let input_dim = 784;
    let mut model = Sequential::new(input_dim.clone(), Cost::MSE);
    model.add(Dense{units: 128, activation: Activations::ReLU, init_func: String::from("he")});
    model.add(Dense{units: 32, activation: Activations::ReLU, init_func: String::from("he")});
    model.add(Dense{units: 10, activation: Activations::Softmax, init_func: String::from("he")});
    model.summary();

    let mut dataset = Vec::new();

    let Mnist { trn_img, trn_lbl, tst_img, tst_lbl, ..} = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(2)
        .validation_set_length(2)
        .test_set_length(2)
        .base_path("/home/tona/Desktop/neuralnets/fe0_ml/src/data")
        .finalize();
 
    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((2, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x: &u8| *x as f32 / 256.0);
    //println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((2, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x: &u8| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    //for i in 0..train_data.shape()[0] {
    for i in 0..1 {
        let image = train_data.slice(s![i, .., ..]);
        let mut label_vec: Array2<f32> = create_weight(vec![1,10]);
        let label_val: usize = train_labels[[i,0]] as usize;
        label_vec[[0, label_val]] = 1f32;
        let sample = Sample{0:flatten(image.to_owned()), 1:label_vec};
        dataset.push(sample);
    }

    let mut output: Array2<f32> = create_weight(vec![1,10]);
    output[[0,0]] = 1f32;
    let epochs = 1;
    let learning_rate = 0.001;
    let first_sample = dataset[0].clone();
    for i in 0..epochs {
        model.train(dataset.clone(), learning_rate);
    }
    model.predict(first_sample.0);
}