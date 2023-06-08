use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Array3, Array2};
use crate::matrixutil::{create_weight, flatten};
use crate::typings::{Dataset, Sample};

pub fn mnist_loader(mut dataset: Dataset, training_samples: usize) -> Dataset{
    let Mnist { trn_img, trn_lbl, ..} = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(training_samples as u32)
        .validation_set_length(1)
        .test_set_length(1)
        .base_path("/home/tona/Desktop/neuralnets/fe0_ml/src/data")
        .finalize();
 
    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((training_samples, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x: &u8| *x as f32 / 256.0);
    //println!("{:#.1?}\n",train_data.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((training_samples, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x: &u8| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    for i in 0..train_data.shape()[0] {
        let image = train_data.slice(s![i, .., ..]);
        let mut label_vec: Array2<f32> = create_weight(&vec![1,10]);
        let label_val: usize = train_labels[[i, 0]] as usize;
        label_vec[[0, label_val]] = 1f32;
        let sample: Sample = Sample{0:flatten(&image.to_owned()), 1:label_vec};
        dataset.push(sample);
    }
    dataset
}