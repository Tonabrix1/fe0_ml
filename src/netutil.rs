use std::ops::{Add,Sub};
use crate::layers::Layers;
use crate::matrixutil::{init_rand, create_weight, init_he, init_xavier, transpose, product};
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

    pub fn train(&mut self, dataset: Vec<Sample>, lr: f32) {
        //TODO: come back and optimize/simplify all this unorganized mess
        let mut x: Array2<f32> = Array2::<f32>::zeros((1,1));
        let mut y: Array2<f32> = Array2::<f32>::zeros((1,1));
        
        //output for collect forward
        let mut fw_vec: Vec<Vec<Array2<f32>>> = Vec::new();

        //vecs for breaking fw_vec back into parts (there's gotta be a better way than this right?)
        let mut a_vec: Vec<Vec<Array2<f32>>> = Vec::new();
        let mut z_vec: Vec<Vec<Array2<f32>>> = Vec::new();

        // stores a_vec and z_vec for backprop and stuff
        let mut predictions: Vec<Vec<Vec<Array2<f32>>>> = Vec::new();
        println!("dataset len: {}", dataset.len());
        for i in 0..dataset.len() {
            x = dataset[i].0.clone();
            y = dataset[i].1.clone();
            if i <= 0 {
                fw_vec = self.collect_forward(x.clone());
                a_vec.push(fw_vec[1].clone());
                z_vec.push(fw_vec[0].clone());
                predictions.push(fw_vec.clone());
                continue;
            }
            fw_vec = self.collect_forward(x.clone());
            
            predictions.push(fw_vec.clone());
            println!("prediction: {:?}\nexpected: {:?}\n\n", fw_vec[1].last().unwrap(), y.clone());
            //println!("z: {:?}\na:{:?}", fw_vec[0][1], fw_vec[1][1]);
        }
        //TODO: what the fuck is this I need to pass multiple x's and y's
        let gradient = Sequential::calculate_gradient(self, predictions, x, vec![y]);
        //println!("Final weight updates: {:?}\nFinal bias updates: {:?}", gradient[0].iter().last().unwrap(), gradient[1][0].iter().last().unwrap());
        //println!("Final weights: {:?}\nFinal bias: {:?}", self.weights.last().unwrap(), self.biases.last().unwrap());
        let last_weight: usize = self.weights.len()-1;
        println!("last_weight: {:?}", last_weight);
        println!("last_gradient: {:?}", gradient[0].len());
        println!("\n\nweights before update: {:?}\n\n", self.weights.last().unwrap());
        for i in (0..last_weight).rev() {
            let new_weight = self.weights[i].clone().sub(gradient[0][i].clone() * lr);
            println!("new weight: {:?}", new_weight);
            self.weights[i] = self.weights[i].clone().sub(gradient[0][i].clone() * lr);
            self.biases[i] = self.biases[i].clone().sub(gradient[1][i].clone() * lr);
        }
        println!("\n\nweights after update: {:?}\n\n", self.weights.last().unwrap());
   }


   //TODO: code input to be Vec<Array2<f32>> for batch
   pub fn calculate_gradient(&self, predictions: Vec<Vec<Vec<Array2<f32>>>>, input: Array2<f32>, expected: Vec<Array2<f32>>) -> Vec<Vec<Array2<f32>>>{
        let batch_size: usize = predictions.len();
        println!("Batch #{}",batch_size);
        let last_pred: usize = predictions[0][0].len();
        println!("Last prediction: {}",last_pred);

        
        let mut raw_grad: Array2<f32> = Array2::<f32>::zeros((1,1));
        let mut weight_updates: Vec<Array2<f32>> = Vec::new();
        let mut bias_updates: Vec<Array2<f32>> = Vec::new();

        //TODO: store transposes of the weights every batch to avoid creating a new transpose array for every batch

        for j in 0..batch_size {
            for i in (0..last_pred).rev() {
                // ∂C/∂w = ∂C/∂A * ∂A/∂Z * ∂Z/∂w  
                if i == last_pred - 1{
                    println!("cost w.r.t a{:?}\n a w.r.t z{:?}", self.cost.derivate(&predictions, &expected), self.layers[i].derivate_activation(predictions[j][0][i].clone()));
                    println!("cost w.r.t z {:?}", self.layers[i].derivate_activation(predictions[j][0][i].clone()) * self.cost.derivate(&predictions, &expected));
                    // ∂C/∂zₙ = ∂C/∂aₙ * ∂aₙ/∂zₙ
                    raw_grad = self.layers[i].derivate_activation(predictions[j][0][i].clone()) * self.cost.derivate(&predictions, &expected);
                    // ∂C/∂wₙ = raw_grad * ∂zₙ/∂wₙ
                    // Z(w,X,b) = w.X + b
                    // ∂Z/∂w = Xᵀ
                    weight_updates.push(transpose(&input).dot(&raw_grad));
                    println!("{:?}", weight_updates);
                } else {
                    println!("a w.r.t z{:?}\nc w.r.t. z[i+1] {:?}", self.layers[i+1].derivate_activation(predictions[j][0][i].clone()), raw_grad.clone());
                    println!("cost w.r.t z {:?}", product(transpose(&self.layers[i].derivate_activation(predictions[j][0][i].clone())),raw_grad.clone()));
                    // ∂C/∂zₙ₋₁ = ∂C/∂zₙ * ∂zₙ/∂aₙ₋₁ * ∂aₙ₋₁/∂zₙ₋₁
                    // ∂zₙ/∂aₙ₋₁ = wₙ.T
                    raw_grad = transpose(&self.weights[i]).dot(&product(transpose(&raw_grad), self.layers[i].derivate_activation(predictions[j][0][i].clone())));
                    // ∂C/∂wₙ₋₁ = ∂C/∂zₙ₋₁ * ∂zₙ₋₁/∂wₙ₋₁
                    weight_updates.push(transpose(&predictions[j][1][i]).dot(&raw_grad));
                    println!("\n\nweight update vec: {:?}\n\n",weight_updates);
                }
                // ∂C/∂bₙ = ∂C/∂A * ∂A/∂Z * ∂Z/∂bₙ
                // ∂Z/∂bₙ = 1
                // ∂C/∂bₙ = raw_grad * 1 = raw_grad
                #[cfg(feature = "a-mode")] {
                    bias_updates.push(raw_grad.clone());
                    println!("\n\nbias update vec: {:?}\n\n", bias_updates);
                }
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