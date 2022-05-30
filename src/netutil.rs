use ndarray::Array2;
use crate::activations::Activation;
use crate::layers::Layer;

// main struct that holds a reference to the layers and biases
// just starting with a sequential model to get everything working
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
    pub biases: Vec<Array2<f32>>,
    pub loss: Option<Box<dyn Loss>>,
}

impl Sequential {

    pub fn new() -> Sequential {
        Sequential {
            layers: Vec::new(),
            biases: Vec::new(),
            loss: None,
        }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
        //generate a bias layer
    }

    pub fn train(
        &mut self,
        mut dataset: Vec<Sample>,
        epochs: i32,
        batch: Option<i32>,
        lr: Option<f32>,
    ) {
        /*
        // learning rate defaults to 0.001
        let lr: f32 = lr.unwrap_or(0.001);
        // batch size
        let batch: i32 = batch.unwrap_or(128);

        // first element of the Sample tuple
        let mut inputs: Array2<f32>;
        // second element of the Sample tuple
        let mut labels: Array2<f32>
        let mut accuracies: Vec<f32> = Vec::new();
        let mut losses: Vec<f32> = Vec::new();
        let mut next_weights: Vec<Array2<f32>>;
        let mut next_biases: Vec<Array2<f32>>;

        for epoch in 0..epochs {
            next_weights = Vec::new();
            next_biases = Vec::new();
            println!("Starting epoch {}", epoch);
            // new sample
            let sample: Sample = dataset.pop().expect("Not enough samples");
            let x: Array2<f32> = sample.0.clone();
            let y: Array2<f32> = sample.1;

            let (fp_acts, fp_dots, error) = forward_propagate(net, x, y);
            let out = *&fp_acts.last().expect("empty activations vector").clone();

            let category = arg_max(out.clone());
            println!("argmax: {:?}", category);
            guess_onehot[[0, category.1.clone()]] = 1.;
            let accuracy = y.clone()[[category.clone().1, 0]];
            accuracies.push(accuracy);
            println!("guess onehot: {:?}", guess_onehot.clone());
            println!("y onehot: {:?}", y.clone());
            let loss = mean_squared_error(guess_onehot.clone(), y.clone()).expect("MSE failed");
            losses.push(loss);

        }*/
    }
}

pub trait Loss {
    fn new(&self) -> Box<dyn Loss>;
}
// holds a single (input : onehot_label) pair,
pub struct Sample(Array2<f32>, Array2<f32>);
