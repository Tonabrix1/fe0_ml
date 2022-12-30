use std::ops::Sub;
use crate::layers::Layers;
use crate::matrixutil::{init_rand, create_layer, init_he, init_xavier};
use crate::cost::{Cost};
use ndarray::{Array2, Ix2};

pub trait Net {
    fn add(&mut self, layer: Layers);
    fn summary(&self);
}

#[derive(Clone)]
pub struct Sample(pub Array2<f32>, pub Array2<f32>);

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
            input_dim: input_dim,
            cost: cost,
        }
    }

    pub fn generate_weights(&mut self, layer: &Layers) {
        let dim: Vec<usize>;
        if self.weights.len() <= 0 { dim = vec![self.input_dim, layer.get_units()]; }
        else { dim = vec![*self.weights.last().unwrap().shape().last().unwrap(), layer.get_units()]; }
        // converting the string back and forth like this is ugly as fuck
        let new_weights: Array2<f32> = match &layer.get_init_func().to_string().to_lowercase()[..] {
            "xavier" | "glorot" => init_xavier(dim.clone()),
            "kaiming" | "he" => init_he(dim.clone()),
            //"lecun" => init_lecun(dim)
            _ => init_rand(dim.clone()),
        };
        self.weights.push(new_weights);
    }


    pub fn generate_biases(&mut self, layer: &Layers) {
        let dim: Vec<usize> = vec![1, (*layer).get_units()];
        let new_bias = create_layer::<Ix2>(dim);
        self.biases.push(new_bias);
    }

    pub fn predict(&self, input: Array2<f32>) -> Array2<f32>{
        let mut x: Array2<f32> = input;
        //doing this dumb shit because rust won't let me run code with values that "could be uninitialized" fuck you
        let mut z: Array2<f32> = self.layers[0].forward_propagate(x.clone(), self.weights[0].clone(), self.biases[0].clone());
        let mut a: Array2<f32> = self.layers[0].activate(z.clone());
        x = a.clone();
        for i in 1..self.layers.len() {
            z = self.layers[i].forward_propagate(x, self.weights[i].clone(), self.biases[i].clone());
            a = self.layers[i].activate(z.clone());
            x = a.clone();
        }
        a
    }

    pub fn train(&mut self, dataset: Vec<Sample>, lr: f32) {
        let mut x: Array2<f32> = dataset[0].0.clone();
        let mut y: Array2<f32> = dataset[0].1.clone();
        let mut fw_vec: Vec<Vec<Array2<f32>>> = Vec::new();
        let mut predictions: Vec<Vec<Vec<Array2<f32>>>> = Vec::new();
        println!("dataset len: {}", dataset.len());
        for i in 0..dataset.len() {
            if i <= 0 {
                fw_vec = self.collect_forward(x.clone());
                
                predictions.push(fw_vec.clone());
                println!("prediction: {:?}\nexpected: {:?}\n\n", fw_vec[1][1], y.clone());
            }
            x = dataset[i].0.clone();
            dataset[i].1.clone();
            fw_vec = self.collect_forward(x.clone());
            
            predictions.push(fw_vec.clone());
            println!("prediction: {:?}\nexpected: {:?}\n\n", fw_vec[1][2], y.clone());
            //println!("z: {:?}\na:{:?}", fw_vec[0][1], fw_vec[1][1]);
        }
        let gradient = Sequential::calculate_gradient(self, fw_vec, x, y);
        println!("Final weight updates: {:?}\nFinal bias updates: {:?}", gradient[0].last().unwrap(), gradient[1].last().unwrap());
        println!("Final weights: {:?}\nFinal bias: {:?}")
        let last_weight = self.weights.len()-1;
        for i in last_weight..=0 {
            self.weights[i] = self.weights[i].clone().sub(gradient[0][last_weight-i].clone() * lr);
            self.biases[i] = self.weights[i].clone().sub(gradient[1][last_weight-i].clone() * lr);
        }
   }

   pub fn calculate_gradient(&self, fw_vec: Vec<Vec<Array2<f32>>>, input: Array2<f32>, expected: Array2<f32>) -> Vec<Vec<Array2<f32>>>{
        //both fw[0] and fw[1] should have equal lengths so this is used to index the last element for both
        let last_pred: usize = fw_vec[0].len() - 1;
        let mut raw_grad: Array2<f32> = self.layers[0].derivate_activation(fw_vec[0][last_pred].clone()) * self.cost.derivate(fw_vec[1][last_pred].clone(), expected.clone());
        let mut weight_updates: Vec<Array2<f32>> = vec![fw_vec[1][last_pred].clone() * raw_grad.clone()];
        let mut bias_updates: Vec<Array2<f32>> = vec![raw_grad.clone()];

        //we loop backwards boyeiiiiiiiiii, starting at the second to last element in the array because stinky rust compiler shenanigans
        for i in last_pred-1..=0 {
            raw_grad = self.weights[i-1].clone() * self.layers[i].derivate_activation(fw_vec[0][i].clone()) * raw_grad.clone();
            bias_updates.push(raw_grad.clone());
            if i > 0{
                weight_updates.push(fw_vec[1][i].clone() * raw_grad.clone());
                continue;
            }
            weight_updates.push(input.clone() * raw_grad.clone());
        }
        vec![weight_updates, bias_updates]
   }

   pub fn collect_forward(&self, input: Array2<f32>) -> Vec<Vec<Array2<f32>>>{
        let mut z_vec = Vec::new();
        let mut a_vec = Vec::new();
        let mut a: Array2<f32>;
        let mut z: Array2<f32>;
        let mut x: Array2<f32> = input;
        // okay this is one of the worst places to have clone statements for array2<f32> types
        // I'm planning to optimize this heavily but I just want a working implementation atm
        for i in 0..self.layers.len() {
            z = self.layers[i].forward_propagate(x, self.weights[i].clone(), self.biases[i].clone());
            z_vec.push(z.clone());
            a = self.layers[i].activate(z);
            x = a.clone();
            a_vec.push(a);
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
            println!("{} / Dimensions: {:?}", self.layers[i].display(), self.weights[i].shape()); 
        }
    }
}