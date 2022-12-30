use ndarray::Array2;
use crate::layers::Layers;
use crate::matrixutil::{init_rand, create_layer, init_he, init_xavier};
use ndarray::Ix2;

pub trait Net {
    fn add(&mut self, layer: Layers);
    fn summary(&self);
}

pub struct Sample(pub Array2<f32>, pub Array2<f32>);

// main struct that holds a reference to the layers and biases
// just starting with a sequential model to get everything working
pub struct Sequential {
    pub layers: Vec<Layers>,
    pub weights: Vec<Array2<f32>>,
    pub biases: Vec<Array2<f32>>,
    pub input_dim: usize,
    pub cost_func: Cost
}

#[allow(dead_code)]
impl Sequential {
    pub fn new(input_dim: usize) -> Self {
        Sequential {
            layers: Vec::new(),
            weights: Vec::new(),
            biases: Vec::new(),
            input_dim: input_dim,
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
        println!("weights: {:?}", new_weights.clone());
        self.weights.push(new_weights);
    }


    pub fn generate_biases(&mut self, layer: &Layers) {
        let dim: Vec<usize> = vec![1, (*layer).get_units()];
        let new_bias = create_layer::<Ix2>(dim);
        println!("bias: {:?}", new_bias.clone());
        self.biases.push(new_bias);
    }

    pub fn predict(&self, input: Array2<f32>) -> Array2<f32>{
        let mut x: Array2<f32> = input;
        let mut z: Array2<f32>;
        let mut a: Array2<f32>;

        for i in 0..self.layers.len() {
            z = self.layers[i].forward_propagate(x, self.weights[i].clone(), self.biases[i].clone());
            a = self.layers[i].activate(z.clone());
            x = a.clone();
        }
    }

    pub fn train(&self, dataset: Vec<Sample>) {
        let mut x: Array2<f32>;
        let mut y: Array2<f32>;
        let fw_vec: Vec<Vec<Array2<f32>>>;
        let predictions: Vec<Array2<f32>>;
        for i in 0..dataset.len() {
            x = dataset[i].0.clone();
            y = dataset[i].1.clone();
            fw_vec = self.collect_forward(x);
            
            predictions.push(fw_vec)
            println!("prediction: {:?}\nexpected: {:?}\n\n", fw_vec[1][1], y);
            //println!("z: {:?}\na:{:?}", fw_vec[0][1], fw_vec[1][1]);
        }
        let gradient = self.calculate_gradient(fw_vec, x, y);
        println!("Weight updates: {:?}\nBias updates: {:?}", gradient.0, gradient.1);
   }

   pub fn calculate_gradient(fw_vec: Vec<Vec<Array2<f32>>>, expected: Vec<Array2<f32>>, input: Array2<f32>, expected: Array2<f32>) -> Vec<Vec<Array2<f32>>>{
        let mut raw_grad: Array2<f32>;
        let mut weight_updates: Array2<f32>;
        let mut bias_updates: Array2<f32>;

        //we loop backwards boyeiiiiiiiiii
        for i in fw_vec.len()-1..=0 {
            //last layer doesn't have any weights between the activation output and the cost function 
            if i >= fw_vec.len() {
                raw_grad = self.layers[i].derivate_activation(fw_vec[i].0) * self.layers[i].cost.derivate(self.layers[i].derivate(fw_vec[i].1, expected[i]));
                weight_updates.push(fw_vec[i].1 * raw_grad);
                bias_updates.push(raw_grad);
                continue;
            }
            raw_grad = self.weights[i-1] * self.layers.derivate_activation(fw_vec[i].0) * raw_grad;
            bias_updates.push(raw_grad);
            if i > 0{
                weight_updates.push(fw_vec[i].1 * raw_grad);
                continue;
            }
            weight_updates.push(input * raw_grad);
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
            a = self.layers[i].activate(z.clone());
            x = a.clone();
            a_vec.push(a.clone());
        }
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