#![allow(dead_code, unused_variables, non_snake_case)]
use crate::{
    cost::Cost,
    layers::Layers,
    matrixutil::{create_weight, init_he, init_rand, init_xavier},
    optimizers::Optimizers,
    typings::{BatchedDataset, Dataset, ForwardBatch},
};
use ndarray::{Array2, Ix2};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::ops::Sub;

pub trait Net {
    fn add(&mut self, layer: Layers);
    fn summary(&self);
}

// main struct that holds a reference to the layers and biases
// just starting with a sequential model to get everything working
pub struct Sequential {
    pub layers: Vec<Layers>,
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array2<f32>>,
    pub input_dim: usize,
    pub cost: Cost,
}

#[allow(dead_code)]
impl Sequential {
    pub fn new(input_dim: usize, cost: Cost) -> Self {
        Sequential {
            layers: Vec::new(),
            weights: Vec::new(),
            biases: Vec::new(),
            input_dim,
            cost,
        }
    }

    pub fn generate_weights(&mut self, layer: &Layers) {
        //possibly wasteful initialization here for the compiler to be comfy
        let dim: &mut Vec<usize> = &mut Vec::with_capacity(2);
        if self.weights.len() <= 0 {
            dim.push(self.input_dim);
            dim.push(layer.get_units());
        } else {
            dim.push(*self.weights.last().unwrap().shape().last().unwrap());
            dim.push(layer.get_units());
        }
        // converting the string back and forth like this is ugly as fuck
        println!("{:?}", dim);
        let new_weights: Array2<f32> = match &layer.get_init_func().to_string().to_lowercase()[..] {
            "xavier" | "glorot" => init_xavier(dim),
            "kaiming" | "he" => init_he(dim),
            //"lecun" => init_lecun(dim)
            _ => init_rand(dim),
        };
        self.weights.push(new_weights);
    }

    pub fn generate_biases(&mut self, layer: &Layers) {
        let new_bias = create_weight::<Ix2>(&vec![1, (*layer).get_units()]);
        self.biases.push(new_bias);
    }

    pub fn predict(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut x: &Array2<f32> = input;
        //doing this dumb shit because rust won't let me run code with values that "could be uninitialized" fuck you
        let mut z: Array2<f32> =
            self.layers[0].forward_propagate(x, &self.weights[0], &self.biases[0]);
        let mut a: Array2<f32> = self.layers[0].activate(&mut z);
        x = &a;
        for i in 1..self.layers.len() {
            z = self.layers[i].forward_propagate(x, &self.weights[i], &self.biases[i]);
            a = self.layers[i].activate(&mut z);
            x = &a;
        }
        a
    }

    pub fn train(
        &mut self,
        dataset: Dataset,
        optimizer: Optimizers,
        lr: f32,
        batch_size: usize,
        epochs: usize,
    ) {
        //TODO: come back and optimize/simplify all this unorganized mess
        let mut x: Array2<f32>;
        let mut y: Array2<f32>;
        let dataset_len = dataset.len() as f32;

        //output for collect forward; wasteful initialization here to keep the compiler comfy, optimize this
        let mut fw_vec: Vec<Vec<Array2<f32>>>;

        // stores a_vec and z_vec for backprop and stuff; wasteful initialization here to keep the compiler comfy, optimize this
        let mut predictions: ForwardBatch = Vec::with_capacity(dataset_len as usize);
        let mut batch_input: Vec<Array2<f32>> = Vec::with_capacity(batch_size);
        let mut batch_labels: Vec<Array2<f32>> = Vec::with_capacity(batch_size);
        //println!("dataset len: {}", dataset.len());
        let batches: BatchedDataset = Self::create_batches(dataset, batch_size);
        for _ in 0..epochs {
            for batch in batches.iter() {
                for sample in batch.iter() {
                    x = sample.0.to_owned();
                    y = sample.1.to_owned();
                    fw_vec = self.collect_forward(&x);
                    predictions.push(fw_vec.to_owned());
                    //println!("prediction: {:?}\nexpected: {:?}\n\n", fw_vec[1].last().unwrap(), y.clone());
                    batch_input.push(x.to_owned());
                    batch_labels.push(y.to_owned());
                }

                println!(
                    "cost: {:?}",
                    self.cost.calculate(&predictions, &batch_labels)
                );

                //TODO: what the fuck is this I need to pass multiple x's and y's
                let gradient =
                    optimizer.backward(&self, &mut predictions, &batch_input, &batch_labels);
                //println!("Final weight updates: {:?}\nFinal bias updates: {:?}", gradient[0].iter().last().unwrap(), gradient[1][0].iter().last().unwrap());
                //println!("Final weights: {:?}\nFinal bias: {:?}", self.weights.last().unwrap(), self.biases.last().unwrap());
                let last_weight: usize = self.weights.len() - 1;
                let last_grad = gradient[0].len() - 1;
                //println!("\n\nweights before update: {:?}\n\n", self.weights.last().unwrap());
                for i in (0..last_weight).rev() {
                    self.weights[i] = self.weights[i]
                        .clone()
                        .sub(gradient[0][last_grad - i].clone() * lr);
                    self.biases[i] = self.biases[i]
                        .clone()
                        .sub(gradient[1][last_grad - i].clone() * lr);
                }
                predictions.clear();
                batch_input.clear();
                batch_labels.clear()
            }
        }
    }

    pub fn create_batches(dataset: Dataset, batch_size: usize) -> BatchedDataset {
        let mut batches: BatchedDataset = Vec::new();
        let mut temp_batches: Dataset = Vec::new();

        let mut batch_indices: Vec<u32> = (0..dataset.len() as u32).collect();
        let mut rng = thread_rng();
        batch_indices.shuffle(&mut rng);
        for i in batch_indices {
            temp_batches.push(dataset[i as usize].clone());
            if temp_batches.len() % batch_size == 0 {
                batches.push(temp_batches);
                temp_batches = Vec::new();
            }
        }
        if temp_batches.len() != 0 {
            batches.push(temp_batches);
        }
        println!("Num Batches Loaded: {}", batches.len());
        batches
    }

    pub fn collect_forward(&self, input: &Array2<f32>) -> Vec<Vec<Array2<f32>>> {
        let mut z_vec = Vec::new();
        let mut a_vec = Vec::new();
        let mut a: Array2<f32>;
        let mut z: Array2<f32>;
        let mut x: &Array2<f32> = input;
        // okay this is one of the worst places to have clone statements for array2<f32> types
        // I'm planning to optimize this heavily but I just want a working implementation atm
        for i in 0..self.layers.len() {
            z = self.layers[i].forward_propagate(x, &self.weights[i], &self.biases[i]);
            a = self.layers[i].activate(&mut z);
            z_vec.push(z.to_owned());
            x = &a;
            a_vec.push(a.to_owned());
        }
        //println!("z: {:?}\n\na: {:?}\n\n", z_vec.clone(), a_vec.clone());
        vec![z_vec, a_vec]
    }
}

impl Net for Sequential {
    fn add(&mut self, layer: Layers) {
        // push the layer to the network's layer vector

        self.generate_weights(&layer);
        self.generate_biases(&layer);

        self.layers.push(layer);
    }

    fn summary(&self) {
        for i in 0..self.layers.len() {
            println!(
                "{} / Dimensions: {:?}",
                self.layers[i].display(),
                self.weights[i].shape()
            );
        }
    }
}
