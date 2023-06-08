use std::ops::Sub;
use crate::layers::Layers;
use crate::matrixutil::{init_rand, create_weight, init_he, init_xavier, transpose};
use rand::thread_rng;
use rand::seq::SliceRandom;
use crate::cost::Cost;
use ndarray::{Array2, Ix2};
use super::typings::{dataset, forward_batch, ds_batch, Sample};

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
        let new_bias = create_weight::<Ix2>(dim);
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

    pub fn train(&mut self, dataset: dataset, lr: f32, batch_size: usize, epochs: usize) {
        //TODO: come back and optimize/simplify all this unorganized mess
        let mut x: Array2<f32> = Array2::<f32>::zeros((1,1));
        let mut y: Array2<f32> = Array2::<f32>::zeros((1,1));
        let dataset_len = dataset.len() as f32;
        
        //output for collect forward; wasteful initialization here to keep the compiler comfy, optimize this
        let mut fw_vec: Vec<Vec<Array2<f32>>> = Vec::new();

        // stores a_vec and z_vec for backprop and stuff; wasteful initialization here to keep the compiler comfy, optimize this
        let mut predictions: forward_batch = Vec::with_capacity(dataset_len as usize);
        let mut batch_input: Vec<Array2<f32>> = Vec::with_capacity(batch_size);
        let mut batch_labels: Vec<Array2<f32>> = Vec::with_capacity(batch_size);
        //println!("dataset len: {}", dataset.len());
        let batches: ds_batch = Self::create_batches(dataset, batch_size);
        for _ in 0..epochs {
            for batch in batches.iter() {
                for sample in batch.iter() {
                    x = sample.0.to_owned();
                    y = sample.1.to_owned();
                    fw_vec = self.collect_forward(x.clone());
                    predictions.push(fw_vec.clone());
                    //println!("prediction: {:?}\nexpected: {:?}\n\n", fw_vec[1].last().unwrap(), y.clone());
                    batch_input.push(x.clone());
                    batch_labels.push(y.clone());
                }
            
                println!("cost: {:?}", self.cost.calculate(&predictions, &batch_labels));

                //TODO: what the fuck is this I need to pass multiple x's and y's
                let gradient = Sequential::calculate_gradient(self, &predictions, &batch_input, &batch_labels);
                //println!("Final weight updates: {:?}\nFinal bias updates: {:?}", gradient[0].iter().last().unwrap(), gradient[1][0].iter().last().unwrap());
                //println!("Final weights: {:?}\nFinal bias: {:?}", self.weights.last().unwrap(), self.biases.last().unwrap());
                let last_weight: usize = self.weights.len()-1;
                let last_grad = gradient[0].len() - 1;
                //println!("\n\nweights before update: {:?}\n\n", self.weights.last().unwrap());
                for i in (0..last_weight).rev() {
                    self.weights[i] = self.weights[i].clone().sub(gradient[0][last_grad - i].clone() * lr);
                    self.biases[i] = self.biases[i].clone().sub(gradient[1][last_grad - i].clone() * lr);
                }
                predictions.clear();
                batch_input.clear();
                batch_labels.clear()
            }
        }
    }


   pub fn create_batches(dataset: dataset, batch_size: usize) -> ds_batch{
        let mut batches:ds_batch = Vec::new();
        let mut temp_batches: dataset = Vec::new();

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

   //TODO: code input to be Vec<Array2<f32>> for batch
   pub fn calculate_gradient(&self, predictions: &forward_batch, input: &Vec<Array2<f32>>, expected: &Vec<Array2<f32>>) -> Vec<Vec<Array2<f32>>>{
        let batch_size: usize = predictions.len();
        let last_pred: usize = predictions[0][0].len();

        
        let mut C_wrt_z: Array2<f32> = Array2::<f32>::zeros((1,1));
        let mut C_wrt_a: Array2<f32> = Array2::<f32>::zeros((1,1));
        let mut weight_updates: Vec<Array2<f32>> = Vec::new();
        let mut bias_updates: Vec<Array2<f32>> = Vec::new();

        //TODO: store transposes of the weights every batch to avoid creating a new transpose array for every batch

        for j in 0..batch_size {
            for i in (0..last_pred).rev() {
                let z = predictions[j][0][i].clone();
                let a_prev = if i > 0 {&predictions[j][1][i-1]} else {&input[j]};
                // ∂C/∂w = ∂Z/∂w * ∂A/∂Z * ∂C/∂A
                if i == last_pred - 1{
                    // ∂C/∂zₙ = ∂aₙ/∂zₙ * ∂C/∂aₙ
                    C_wrt_z = self.layers[i].derivate_activation(z) * self.cost.derivate(&predictions, &expected);
                } else {
                    // ∂C/∂aₙ₋₁ = ∂zₙ/∂aₙ₋₁ * ∂C/∂zₙ
                    // ∂zₙ/∂aₙ₋₁ = wₙ.T
                    //println!("{:?} x {:?}",C_wrt_z.shape(), transpose(&self.weights[i+1]).shape());
                    C_wrt_a = C_wrt_z.dot(&transpose(&self.weights[i+1]));
                    // ∂C/∂zₙ₋₁ = ∂aₙ₋₁/∂zₙ₋₁ * ∂C/∂aₙ₋₁
                    //println!("{:?} * {:?}",C_wrt_a.shape(), self.layers[i].derivate_activation(z.clone()).shape());
                    C_wrt_z = self.layers[i].derivate_activation(z) * C_wrt_a;
                }
                // ∂C/∂bₙ = ∂Z/∂bₙ * ∂A/∂Z * ∂C/∂A
                // ∂Z/∂bₙ = 1
                // ∂C/∂bₙ = 1 * C_wrt_z = C_wrt_z
                bias_updates.push(C_wrt_z.clone());
                // ∂C/∂wₙ = ∂zₙ/∂wₙ * C_wrt_z
                // Z(w,X,b) = w.X + b
                // ∂Z/∂w = Xᵀ
                //println!("{:?} x {:?}",transpose(a_prev).shape(), C_wrt_z.shape());
                weight_updates.push(transpose(a_prev).dot(&C_wrt_z));
                //println!("\n\nweights update dim: {:?}\n\n", transpose(a_prev).dot(&C_wrt_z).shape());
            }
        }
        //println!("\n\nweight update vec: {:?}\n\n",weight_updates);
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